use anyhow::Result;
use candle::quantized::{gguf_file, GgmlDType, QTensor};
use candle::Device;
use std::collections::HashMap;

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
