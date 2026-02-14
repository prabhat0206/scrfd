# Changelog

## [1.3.0] - 2026-02-14

### New Features

- **Builder Pattern**: Introduced `SCRFDBuilder` for fluent and simplified model configuration.
- **Async Support**: Added full asynchronous capabilities via `tokio` with `SCRFDAsync`, enabling non-blocking face detection.
- **Execution Providers**: Added support for various ONNX Runtime execution providers including CUDA, CoreML, DirectML, OpenVINO, TensorRT, ROCm, and more.

### Improvements

- **Performance**: Optimized array operations using `mapv_inplace()` to reduce memory allocation overhead.
- **Error Handling**: Enhanced error reporting in model execution and session creation.
- **Compatibility**: Adjusted method signatures and internal logic to handle breaking changes in `ort` crate.

### Dependencies

- Updated `ort` to `2.0.0-rc.11`.
- Updated `opencv` to `0.98.1`.
- Updated `anyhow`, `uuid`, and `ndarray` dependencies.

### Internal

- Applied `cargo fmt` for consistent code style.
- identifying and removing unused code.

## [1.2.0] - 2025-05-19

### Breaking Changes

- Changed `detect` function to accept `opencv::core::Mat` instead of `image::RgbImage`
  - Significant performance improvements through:
    - Elimination of unnecessary image conversions
    - Optimized memory management with OpenCV
    - Hardware-accelerated image processing operations
    - Reduced memory allocations and copies
    - Direct compatibility with OpenCV's ecosystem

### New Features

- Introduced Builder Pattern for model configuration
  - New `SCRFDBuilder` for fluent interface
  - Simplified model initialization
  - Better configuration management
- Added support for relative output coordinates
- Enhanced documentation with comprehensive examples
- Improved error handling and type safety

### Parameter Updates

- Default confidence threshold changed from 0.5 to 0.25
- Default IoU threshold changed from 0.5 to 0.4

### Documentation

- Added detailed examples for both synchronous and asynchronous usage
- Updated readme with breaking changes section
- Enhanced API documentation with builder pattern examples
- Added comprehensive usage examples with OpenCV integration

### Internal Changes

- Refactored image processing pipeline
- Optimized input preparation using OpenCV
- Removed deprecated helper functions
- Updated package metadata for crates.io publishing

### Dependencies

- Updated OpenCV integration to version 0.93.4
- ONNX Runtime updated to version 2.0.0-rc.9
