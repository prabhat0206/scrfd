# SCRFD - Rust Package for Face Detection

**SCRFD** is a Rust library for face detection, providing both **synchronous** and **asynchronous** support. It utilizes ONNX Runtime for high-performance inference and supports bounding box and keypoint detection.

---

## Breaking Changes in v2.0.0

- Removed crate re-exports of `ort`, `opencv`, and `ndarray`
  - Users must now add these dependencies directly to their `Cargo.toml`
  - This does NOT affect users who already import these crates directly
  - See Installation section for compatible versions

---

## Breaking Changes in v1.2.0

- The `detect` function now accepts `opencv::core::Mat` instead of `image::RgbImage`
  - This change significantly improves performance by:
    - Eliminating unnecessary image conversions
    - Using OpenCV's optimized memory management
    - Leveraging hardware-accelerated image processing operations
    - Reducing memory allocations and copies
    - Providing direct compatibility with OpenCV's extensive computer vision ecosystem
- Added builder pattern for easier model configuration
- Added support for relative output coordinates
- Updated default parameters:
  - Confidence threshold changed from 0.5 to 0.25
  - IoU threshold changed from 0.5 to 0.4

---

## Features
- **Face Detection**: Detect bounding boxes and landmarks for faces in images.
- **Asynchronous Support**: Optional async functionality for non-blocking operations.
- **Builder Pattern**: Fluent interface for easy model configuration.
- **Customizable Parameters**:
  - Input size
  - Confidence threshold
  - IoU threshold
  - Relative output coordinates
- **Efficient Processing**:
  - Anchor-based detection
  - Optimized caching for anchor centers

---

## Installation

Add the library and its required peer dependencies to your `Cargo.toml`:
```toml
[dependencies]
rusty_scrfd = { version = "2.0.0", features = ["async"] } # Enable async feature if needed
ort = { version = "=2.0.0-rc.13", features = ["ndarray"] }
opencv = "0.100"
ndarray = "0.17"
```

To enable synchronous mode only, omit the `async` feature:
```toml
[dependencies]
rusty_scrfd = "2.0.0"
ort = { version = "=2.0.0-rc.13", features = ["ndarray"] }
opencv = "0.100"
ndarray = "0.17"
```

> **Note:** `ort`, `opencv`, and `ndarray` types appear in `rusty_scrfd`'s public API. You must use compatible versions to avoid type mismatches at compile time.

---

## Usage

### Using the Builder Pattern (Recommended)

```rust
use rusty_scrfd::builder::SCRFDBuilder;
use ort::session::SessionBuilder;
use std::collections::HashMap;
use opencv::imgcodecs::imread;
use opencv::imgcodecs::IMREAD_COLOR;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load the ONNX model
    let model_path = "path/to/scrfd_model.onnx";
    let session = SessionBuilder::new().unwrap().with_model_from_file(model_path)?;

    // Initialize SCRFD using the builder pattern
    let mut scrfd = SCRFDBuilder::new(session)
        .set_input_size((640, 640))
        .set_conf_thres(0.25)
        .set_iou_thres(0.4)
        .set_relative_output(true)
        .build()?;

    // Load an image using OpenCV
    let image = imread("path/to/image.jpg", IMREAD_COLOR)?;

    // Center cache to optimize anchor generation
    let mut center_cache = HashMap::new();

    // Detect faces
    let (bboxes, keypoints) = scrfd.detect(&image, 5, "max", &mut center_cache)?;

    println!("Bounding boxes: {:?}", bboxes);
    if let Some(kps) = keypoints {
        println!("Keypoints: {:?}", kps);
    }

    Ok(())
}
```

### Direct Initialization

```rust
use rusty_scrfd::SCRFD;
use ort::session::SessionBuilder;
use std::collections::HashMap;
use opencv::imgcodecs::imread;
use opencv::imgcodecs::IMREAD_COLOR;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load the ONNX model
    let model_path = "path/to/scrfd_model.onnx";
    let session = SessionBuilder::new().unwrap().with_model_from_file(model_path)?;

    // Initialize SCRFD
    let mut scrfd = SCRFD::new(
        session,
        (640, 640),  // input size
        0.25,        // confidence threshold
        0.4,         // IoU threshold
        true         // relative output
    )?;

    // Load an image using OpenCV
    let image = imread("path/to/image.jpg", IMREAD_COLOR)?;

    // Center cache to optimize anchor generation
    let mut center_cache = HashMap::new();

    // Detect faces
    let (bboxes, keypoints) = scrfd.detect(&image, 5, "max", &mut center_cache)?;

    println!("Bounding boxes: {:?}", bboxes);
    if let Some(kps) = keypoints {
        println!("Keypoints: {:?}", kps);
    }

    Ok(())
}
```

---

### Asynchronous Example
Enable the `async` feature in `Cargo.toml`:
```toml
[dependencies]
rusty_scrfd = { version = "2.0.0", features = ["async"] }
```

```rust
use rusty_scrfd::builder::SCRFDBuilder;
use ort::session::SessionBuilder;
use std::collections::HashMap;
use opencv::imgcodecs::imread;
use opencv::imgcodecs::IMREAD_COLOR;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load the ONNX model
    let model_path = "path/to/scrfd_model.onnx";
    let session = SessionBuilder::new().unwrap().with_model_from_file(model_path)?;

    // Initialize SCRFDAsync using the builder pattern
    let scrfd = SCRFDBuilder::new(session)
        .set_input_size((640, 640))
        .set_conf_thres(0.25)
        .set_iou_thres(0.4)
        .set_relative_output(true)
        .build_async()?;

    // Load an image using OpenCV
    let image = imread("path/to/image.jpg", IMREAD_COLOR)?;

    // Center cache to optimize anchor generation
    let mut center_cache = HashMap::new();

    // Detect faces asynchronously
    let (bboxes, keypoints) = scrfd.detect(&image, 5, "max", &mut center_cache).await?;

    println!("Bounding boxes: {:?}", bboxes);
    if let Some(kps) = keypoints {
        println!("Keypoints: {:?}", kps);
    }

    Ok(())
}
```

---

## API Documentation

### Builder Pattern
The `SCRFDBuilder` provides a fluent interface for configuring SCRFD models:

```rust
let model = SCRFDBuilder::new(session)
    .set_input_size((640, 640))    // Set input dimensions
    .set_conf_thres(0.25)          // Set confidence threshold
    .set_iou_thres(0.4)            // Set IoU threshold
    .set_relative_output(true)      // Enable relative output
    .build()?;                      // Build synchronous model
```

For async models:
```rust
let model = SCRFDBuilder::new(session)
    .set_input_size((640, 640))
    .set_conf_thres(0.25)
    .set_iou_thres(0.4)
    .set_relative_output(true)
    .build_async()?;               // Build asynchronous model
```

### Default Parameters
- Input size: (640, 640)
- Confidence threshold: 0.25
- IoU threshold: 0.4
- Relative output: true

### Detect Function
The `detect` function now accepts OpenCV's `Mat` type instead of `image::RgbImage`:

```rust
pub fn detect(
    &mut self,
    image: &Mat,                    // OpenCV Mat instead of RgbImage
    max_num: usize,
    metric: &str,
    center_cache: &mut HashMap<(i32, i32, i32), Array2<f32>>,
) -> Result<(Array2<f32>, Option<Array3<f32>>), Box<dyn Error>>
```

### Helper Functions
**Available in `ScrfdHelpers`**:
- `generate_anchor_centers`: Efficiently generate anchor centers for feature maps
- `distance2bbox`: Convert distances to bounding boxes
- `distance2kps`: Convert distances to keypoints
- `nms`: Perform non-maximum suppression to filter detections

---

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for improvements.

### Running Tests
- For synchronous features:
  ```bash
  cargo test
  ```
- For asynchronous features:
  ```bash
  cargo test --features async
  ```

---

## License
This library is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## References
- [SCRFD Paper](https://arxiv.org/abs/2105.04714)
- [ONNX Runtime](https://onnxruntime.ai/)

