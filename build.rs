fn main() {
    // Tell whisper-rs where to find CUDA on Arch Linux
    println!("cargo:rustc-env=CUDA_PATH=/opt/cuda");
    println!("cargo:rustc-link-search=/opt/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");
}
