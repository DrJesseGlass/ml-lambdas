use anyhow::Result;
use candle::quantized::gguf_file;
use candle::{Device, Tensor};
use candle_transformers::models::quantized_qwen3::ModelWeights as Qwen3;
use std::io::Cursor;

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
    // Reset the high-water mark to the current level, to isolate the next phase.
    pub fn reset_peak() {
        PEAK.store(CURRENT.load(Ordering::Relaxed), Ordering::Relaxed);
    }
}

#[global_allocator]
static ALLOC: track::Tracking = track::Tracking;

fn line(label: &str) {
    println!(
        "  {label:<34} current {:>8.1} MB   peak {:>8.1} MB",
        track::current_mb(),
        track::peak_mb()
    );
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        anyhow::bail!("usage: mem-profile <model.gguf> [prompt_len] [decode_steps]");
    }
    let path = &args[1];
    let prompt_len: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(64);
    let decode_steps: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(64);
    let device = Device::Cpu;

    println!("== LOAD (all-at-once: whole file buffer + from_gguf) ==");
    let buf = std::fs::read(path)?;
    line("after read file -> Vec<u8>");

    let mut model = {
        let mut cursor = Cursor::new(buf);
        let content = gguf_file::Content::read(&mut cursor)?;
        let m = Qwen3::from_gguf(content, &mut cursor, &device)?;
        line("after from_gguf (buffer still held)");
        m
        // cursor (owning the input buffer) drops here
    };
    line("after input buffer freed (resident)");
    let resident_after_load = track::current_mb();

    println!("\n== PREFILL forward ({prompt_len} tokens) ==");
    track::reset_peak();
    let ids: Vec<u32> = (0..prompt_len as u32).map(|i| (i % 1000) + 1).collect();
    let input = Tensor::new(ids.as_slice(), &device)?.unsqueeze(0)?;
    let _logits = model.forward(&input, 0)?;
    line("after prefill forward");
    let prefill_peak = track::peak_mb();

    println!("\n== DECODE forward ({decode_steps} single-token steps) ==");
    track::reset_peak();
    for i in 0..decode_steps {
        let tok = Tensor::new(&[((i as u32) % 1000) + 1], &device)?.unsqueeze(0)?;
        let _logits = model.forward(&tok, prompt_len + i)?;
    }
    line("after decode steps");
    let decode_peak = track::peak_mb();

    let final_current = track::current_mb();
    println!("\n== SUMMARY ==");
    println!("  resident after load (weights + KV-init + RoPE + tables): {resident_after_load:.1} MB");
    println!("  prefill peak transient (over resident): {:.1} MB", prefill_peak - resident_after_load);
    println!("  decode peak transient (over during-decode resident): {:.1} MB", decode_peak - final_current);
    println!("  final current (resident + grown KV after decode): {final_current:.1} MB");
    println!("  global high-water (~= wasm linear memory): {:.1} MB", track::peak_mb());
    Ok(())
}
