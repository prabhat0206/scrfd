pub mod builder;
pub mod helpers;
pub mod scrfd;

#[cfg(feature = "async")]
pub mod scrfd_async;

pub use builder::SCRFDBuilder;
pub use helpers::*;
pub use ndarray;
pub use opencv;
pub use ort;
pub use scrfd::SCRFD;

#[cfg(feature = "async")]
pub use scrfd_async::SCRFDAsync;

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::Vector;
    use opencv::imgcodecs::imdecode;
    use opencv::prelude::MatTraitConst;
    use opencv::{core, imgcodecs, imgproc};
    use ort::execution_providers::CPUExecutionProvider;
    use ort::session::Session;
    use std::collections::HashMap;
    use std::path::Path;

    #[test]
    fn test_sync_face_detection() -> Result<(), Box<dyn std::error::Error>> {
        // Initialize SCRFD
        let model_path = Path::new("models/det_10g.onnx");
        let session = Session::builder()?
            .with_execution_providers([CPUExecutionProvider::default().build()])?
            .commit_from_file(model_path)?;

        let mut scrfd = builder::SCRFDBuilder::new(session)
            .set_input_size((640, 640))
            .set_conf_thres(0.25)
            .set_iou_thres(0.4)
            .set_relative_output(true)
            .build()?;

        // Load test image
        let image_path = "sample_input/1.png";
        let image = std::fs::read(image_path)?;

        let mut image = match imdecode(
            &Vector::<u8>::from_slice(&image),
            opencv::imgcodecs::IMREAD_COLOR,
        ) {
            Ok(img) => img,
            Err(_) => return Err("Failed to decode image".into()),
        };
        // Detect faces
        let mut center_cache = HashMap::new();
        let (detections, _) = scrfd.detect(&image, 0, "max", &mut center_cache)?;

        log::trace!("Detections: {:?}", detections);

        // Draw rectangles around detected faces
        for det in detections.rows() {
            let image_width = image.cols();
            let image_height = image.rows();
            log::trace!("Image size: {}x{}", image_width, image_height);

            let x = det[0];
            let y = det[1];
            let width = det[2];
            let height = det[3];

            let left = (x * image_width as f32) as i32;
            let top = (y * image_height as f32) as i32;
            let mut width = (width * image_width as f32) as i32;
            let mut height = (height * image_height as f32) as i32;

            let offset_width = (width * 5) / 100;
            let offset_height = (height * 5) / 100;
            let left = left.saturating_sub(offset_width);
            let top = top.saturating_sub(offset_height);
            width += offset_width * 2;
            height += offset_height * 2;

            let rect = core::Rect::new(left, top, width, height);
            imgproc::rectangle(
                &mut image,
                rect,
                core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                2,
                imgproc::LINE_8,
                0,
            )?;
        }

        // Save the result
        imgcodecs::imwrite(
            "sample_output/output_sync.jpg",
            &image,
            &core::Vector::new(),
        )?;

        // Verify that we detected at least one face
        assert!(
            detections.nrows() > 0,
            "No faces detected in the test image"
        );

        Ok(())
    }
    #[cfg(feature = "async")]
    #[tokio::test]
    async fn test_async_face_detection() -> Result<(), Box<dyn std::error::Error>> {
        // Initialize SCRFD
        let model_path = Path::new("models/det_10g.onnx");
        let session = Session::builder()?
            .with_execution_providers([CPUExecutionProvider::default().build()])?
            .commit_from_file(model_path)?;

        let mut scrfd = SCRFDAsync::new(session, (640, 640), 0.25, 0.4, true)?;

        // Load test image
        let image_path = "sample_input/1.png";
        let image = std::fs::read(image_path)?;

        let mut image = match imdecode(
            &Vector::<u8>::from_slice(&image),
            opencv::imgcodecs::IMREAD_COLOR,
        ) {
            Ok(img) => img,
            Err(_) => return Err("Failed to decode image".into()),
        };
        // Detect faces
        let mut center_cache = HashMap::new();
        let (detections, _) = scrfd.detect(&image, 0, "max", &mut center_cache).await?;

        log::trace!("Async Detections: {:?}", detections);

        // Draw rectangles around detected faces
        for det in detections.rows() {
            let image_width = image.cols();
            let image_height = image.rows();
            log::trace!("Image size: {}x{}", image_width, image_height);

            let x = det[0];
            let y = det[1];
            let width = det[2];
            let height = det[3];

            let left = (x * image_width as f32) as i32;
            let top = (y * image_height as f32) as i32;
            let mut width = (width * image_width as f32) as i32;
            let mut height = (height * image_height as f32) as i32;

            let offset_width = (width * 5) / 100;
            let offset_height = (height * 5) / 100;
            let left = left.saturating_sub(offset_width);
            let top = top.saturating_sub(offset_height);
            width += offset_width * 2;
            height += offset_height * 2;

            let rect = core::Rect::new(left, top, width, height);
            imgproc::rectangle(
                &mut image,
                rect,
                core::Scalar::new(0.0, 255.0, 0.0, 0.0),
                2,
                imgproc::LINE_8,
                0,
            )?;
        }

        // Save the result
        imgcodecs::imwrite(
            "sample_output/output_async.jpg",
            &image,
            &core::Vector::new(),
        )?;

        // Verify that we detected at least one face
        assert!(
            detections.nrows() > 0,
            "No faces detected in the test image (async)"
        );

        Ok(())
    }
}
