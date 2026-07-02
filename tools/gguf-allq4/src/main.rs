use anyhow::Result;
use candle::quantized::{gguf_file, GgmlDType, QTensor};
use candle::Device;
use std::collections::HashMap;
use std::io::Cursor;

// Heap tracker so --profile can report peak vs steady-state allocation, mirroring
// the wasm load where the input buffer and the built QTensors coexist.
mod track {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicUsize, Ordering};

    static CURRENT: AtomicUsize = AtomicUsize::new(0);
    static PEAK: AtomicUsize = AtomicUsize::new(0);

    pub struct Tracking;
    unsafe impl GlobalAlloc for Tracking {
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {
            let p = System.alloc(l);
            if !p.is_null() {
                let c = CURRENT.fetch_add(l.size(), Ordering::Relaxed) + l.size();
                PEAK.fetch_max(c, Ordering::Relaxed);
            }
            p
        }
        unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
            System.dealloc(p, l);
            CURRENT.fetch_sub(l.size(), Ordering::Relaxed);
        }
    }

    pub fn current_mb() -> f64 {
        CURRENT.load(Ordering::Relaxed) as f64 / 1_048_576.0
    }
    pub fn peak_mb() -> f64 {
        PEAK.load(Ordering::Relaxed) as f64 / 1_048_576.0
    }
}

#[global_allocator]
static ALLOC: track::Tracking = track::Tracking;

fn profile(input: &str) -> Result<()> {
    let buf = std::fs::read(input)?;
    let file_mb = buf.len() as f64 / 1_048_576.0;
    println!(
        "gguf read into buffer: {file_mb:.1} MB (heap now {:.1} MB)",
        track::current_mb()
    );

    let mut cursor = Cursor::new(&buf[..]);
    let content = gguf_file::Content::read(&mut cursor).map_err(|e| e.with_path(input))?;

    // Build every QTensor while the input buffer is still alive -- this is the
    // exact coexistence that drives the wasm peak.
    let mut tensors: Vec<QTensor> = Vec::with_capacity(content.tensor_infos.len());
    let device = Device::Cpu;
    for name in content.tensor_infos.keys() {
        tensors.push(content.tensor(&mut cursor, name, &device)?);
    }
    let peak = track::peak_mb();
    let with_buffer = track::current_mb();
    println!("after building all QTensors (buffer + QTensors alive):");
    println!("  current heap: {with_buffer:.1} MB   peak: {peak:.1} MB");

    drop(buf);
    let steady = track::current_mb();
    println!("after dropping input buffer (QTensors only): {steady:.1} MB");
    println!(
        "=> input buffer held {file_mb:.1} MB on top of {steady:.1} MB of QTensors; \
         peak doubling ~= {:.1} MB",
        with_buffer - steady
    );

    // Streamability: is the file laid out in the order from_gguf reads tensors?
    let mut by_offset: Vec<(&str, u64)> = content
        .tensor_infos
        .iter()
        .map(|(n, i)| (n.as_str(), i.offset))
        .collect();
    by_offset.sort_by_key(|(_, off)| *off);
    println!("\nfirst 12 tensors in FILE-OFFSET order:");
    for (n, off) in by_offset.iter().take(12) {
        println!("  {off:>12}  {n}");
    }
    let embd_pos = by_offset
        .iter()
        .position(|(n, _)| *n == "token_embd.weight");
    println!(
        "token_embd.weight is at file-order index {embd_pos:?} of {}",
        by_offset.len()
    );
    std::hint::black_box(&tensors);
    Ok(())
}

// Streaming load: read tensors in file-offset order straight from a File reader
// (no full in-memory buffer), mirroring building QTensors from a network stream
// as bytes arrive. Peak should be ~= the QTensors alone plus one in-flight tensor.
fn profile_stream(input: &str) -> Result<()> {
    let mut f = std::fs::File::open(input)?;
    let content = gguf_file::Content::read(&mut f).map_err(|e| e.with_path(input))?;
    println!(
        "header parsed from File (heap now {:.1} MB)",
        track::current_mb()
    );

    let mut by_offset: Vec<&str> = content.tensor_infos.keys().map(|s| s.as_str()).collect();
    by_offset.sort_by_key(|n| content.tensor_infos[*n].offset);

    let device = Device::Cpu;
    let mut tensors: HashMap<String, QTensor> = HashMap::with_capacity(by_offset.len());
    for name in by_offset {
        let qt = content.tensor(&mut f, name, &device)?;
        tensors.insert(name.to_string(), qt);
    }
    println!("after streaming all QTensors from File (offset order):");
    println!(
        "  current heap: {:.1} MB   peak: {:.1} MB",
        track::current_mb(),
        track::peak_mb()
    );
    println!("(no 325 MB input buffer was ever held -- this is the streaming target)");
    std::hint::black_box(&tensors);
    Ok(())
}

fn load_gguf(path: &str) -> Result<gguf_file::Content> {
    let mut f = std::fs::File::open(path)?;
    Ok(gguf_file::Content::read(&mut f).map_err(|e| e.with_path(path))?)
}

fn dump(input: &str) -> Result<()> {
    let content = load_gguf(input)?;
    if let Some(ft) = content.metadata.get("general.file_type") {
        println!("metadata general.file_type   = {ft:?}");
    }
    if let Some(qb) = content.metadata.get("general.quantized_by") {
        println!("metadata general.quantized_by = {qb:?}");
    }
    let mut hist: HashMap<GgmlDType, usize> = HashMap::new();
    let mut non_q4: Vec<(&str, GgmlDType)> = Vec::new();
    for (name, info) in &content.tensor_infos {
        *hist.entry(info.ggml_dtype).or_default() += 1;
        if info.ggml_dtype != GgmlDType::Q4K {
            non_q4.push((name.as_str(), info.ggml_dtype));
        }
    }
    let mut hist: Vec<_> = hist.into_iter().collect();
    hist.sort_by_key(|(dt, _)| format!("{dt:?}"));
    println!("tensor dtype histogram:");
    for (dt, n) in &hist {
        println!("  {:>6}: {n}", format!("{dt:?}"));
    }
    println!("non-Q4_K tensors ({}):", non_q4.len());
    for (name, dt) in &non_q4 {
        println!("  {:>6}  {name}", format!("{dt:?}"));
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 3 && args[1] == "--profile" {
        return profile(&args[2]);
    }
    if args.len() == 3 && args[1] == "--profile-stream" {
        return profile_stream(&args[2]);
    }
    if args.len() == 2 {
        return dump(&args[1]);
    }
    if args.len() < 3 {
        anyhow::bail!("usage: gguf-allq4 <input.gguf> <output.gguf>   |   gguf-allq4 <input.gguf>  (dump dtypes)");
    }
    let (input, output) = (&args[1], &args[2]);
    let device = Device::Cpu;

    let mut f = std::fs::File::open(input)?;
    let content = gguf_file::Content::read(&mut f).map_err(|e| e.with_path(input))?;
    let mut owned: Vec<(String, QTensor)> = Vec::with_capacity(content.tensor_infos.len());
    let (mut before, mut after) = (0usize, 0usize);
    for name in content.tensor_infos.keys() {
        let qt = content.tensor(&mut f, name, &device)?;
        before += qt.storage_size_in_bytes();
        let dims = qt.shape().dims();
        let ndim = dims.len();
        let last_dim = dims.last().copied().unwrap_or(0);
        // 1-D tensors (norms/scales) stay at source precision; Q4_K needs the
        // row width divisible by 256.
        let new_qt = if ndim == 1 || qt.dtype() == GgmlDType::Q4K {
            qt
        } else if last_dim % 256 != 0 {
            println!(
                "  skip {name}: last dim {last_dim} not %256, keeping {:?}",
                qt.dtype()
            );
            qt
        } else {
            println!("  {name}: {:?} -> Q4K", qt.dtype());
            let de = qt.dequantize(&device)?;
            QTensor::quantize(&de, GgmlDType::Q4K)?
        };
        after += new_qt.storage_size_in_bytes();
        owned.push((name.clone(), new_qt));
    }

    // Output is all-Q4_K: stamp file_type 14 (Q4_K_S) and credit the requant,
    // overriding the inherited source values.
    let q4k_ftype = gguf_file::Value::U32(14);
    let quantized_by = gguf_file::Value::String("DrJesseGlass".to_string());
    let mut metadata: Vec<(&str, &gguf_file::Value)> = content
        .metadata
        .iter()
        .map(|(k, v)| match k.as_str() {
            "general.file_type" => (k.as_str(), &q4k_ftype),
            "general.quantized_by" => (k.as_str(), &quantized_by),
            _ => (k.as_str(), v),
        })
        .collect();
    if !content.metadata.contains_key("general.file_type") {
        metadata.push(("general.file_type", &q4k_ftype));
    }
    if !content.metadata.contains_key("general.quantized_by") {
        metadata.push(("general.quantized_by", &quantized_by));
    }
    let tensors: Vec<(&str, &QTensor)> = owned.iter().map(|(n, t)| (n.as_str(), t)).collect();
    let mut w = std::fs::File::create(output)?;
    gguf_file::write(&mut w, &metadata, &tensors)?;
    println!(
        "wrote {output}  ({:.1}MB -> {:.1}MB weights)",
        before as f64 / 1e6,
        after as f64 / 1e6
    );
    Ok(())
}
