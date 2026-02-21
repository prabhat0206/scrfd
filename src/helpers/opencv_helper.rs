//! OpenCV Helper Module
//!
//! This module provides a set of helper functions for common OpenCV operations used in face detection.
//! It encapsulates image preprocessing, resizing, and tensor conversion operations in a reusable way.
//!
//! # Examples
//!
//! ```rust
//! use rusty_scrfd::helpers::opencv_helper::OpenCVHelper;
//! use opencv::core::Mat;
//! use opencv::imgcodecs;
//!
//! // Create a new helper instance
//! let helper = OpenCVHelper::new(127.5, 128.0);
//!
//! // Load an image
//! let image = imgcodecs::imread("sample_input/1.png", imgcodecs::IMREAD_COLOR)?;
//!
//! // Use the helper for image processing
//! let (resized_image, scale) = helper.resize_with_aspect_ratio(&image, (640, 640))?;
//! let input_tensor = helper.prepare_input_tensor(&resized_image, (640, 640))?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use ndarray::Array4;
use opencv::{core, dnn, prelude::*};
use std::error::Error;

/// A helper struct for common OpenCV operations
///
/// This struct provides methods for image preprocessing and tensor conversion
/// commonly used in face detection tasks. It maintains normalization parameters
/// (mean and standard deviation) used for image preprocessing.
///
/// # Fields
///
/// * `mean` - The mean value used for image normalization (default: 127.5)
/// * `std` - The standard deviation used for image normalization (default: 128.0)
///
/// # Examples
///
/// ```rust
/// use rusty_scrfd::helpers::opencv_helper::OpenCVHelper;
///
/// let helper = OpenCVHelper::new(127.5, 128.0);
/// ```
pub struct OpenCVHelper {
    mean: f32,
    std: f32,
}

impl OpenCVHelper {
    /// Creates a new instance of OpenCVHelper
    ///
    /// # Arguments
    ///
    /// * `mean` - The mean value for image normalization
    /// * `std` - The standard deviation for image normalization
    ///
    /// # Returns
    ///
    /// A new instance of OpenCVHelper
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rusty_scrfd::helpers::opencv_helper::OpenCVHelper;
    ///
    /// let helper = OpenCVHelper::new(127.5, 128.0);
    /// ```
    pub fn new(mean: f32, std: f32) -> Self {
        Self { mean, std }
    }

    /// Prepares an input tensor from an OpenCV Mat for model inference
    ///
    /// This method performs the following operations:
    /// 1. Normalizes the image using mean and standard deviation
    /// 2. Resizes the image to the target dimensions
    /// 3. Converts the OpenCV Mat to a format suitable for model input
    ///
    /// # Arguments
    ///
    /// * `image` - The input image as an OpenCV Mat
    /// * `input_size` - A tuple of (width, height) specifying the target dimensions
    ///
    /// # Returns
    ///
    /// A Result containing either:
    /// * `Ok(Array4<f32>)` - The prepared input tensor
    /// * `Err(Box<dyn Error>)` - An error if the operation fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rusty_scrfd::helpers::opencv_helper::OpenCVHelper;
    /// use opencv::core::Mat;
    /// use opencv::imgcodecs;
    ///
    /// let helper = OpenCVHelper::new(127.5, 128.0);
    /// let image = imgcodecs::imread("sample_input/1.png", imgcodecs::IMREAD_COLOR)?;
    /// let input_tensor = helper.prepare_input_tensor(&image, (640, 640))?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn prepare_input_tensor(
        &self,
        image: &Mat,
        input_size: (i32, i32),
    ) -> Result<Array4<f32>, Box<dyn Error>> {
        // Preprocess the image using blobFromImage
        let blob = dnn::blob_from_image(
            &image,
            1.0 / self.std as f64,
            core::Size::new(input_size.0 as i32, input_size.1 as i32),
            core::Scalar::new(self.mean as f64, self.mean as f64, self.mean as f64, 0.0),
            true,
            false,
            core::CV_32F,
        )?;

        // Convert OpenCV Mat (CHW format) to ndarray
        let tensor_shape = (1, 3, input_size.1 as usize, input_size.0 as usize);
        let tensor_data: Vec<f32> = blob.data_typed()?.to_vec(); // Convert slice to Vec
        let input_tensor = Array4::from_shape_vec(tensor_shape, tensor_data)?;

        Ok(input_tensor)
    }

    /// Resizes an image while preserving its aspect ratio
    ///
    /// This method performs the following operations:
    /// 1. Calculates new dimensions that preserve the aspect ratio
    /// 2. Resizes the image to fit within the target dimensions
    /// 3. Pads the image if necessary to match the target dimensions
    ///
    /// # Arguments
    ///
    /// * `image` - The input image as an OpenCV Mat
    /// * `target_size` - A tuple of (width, height) specifying the target dimensions
    ///
    /// # Returns
    ///
    /// A Result containing either:
    /// * `Ok((Mat, f32, i32, i32))` - A tuple containing:
    ///   - The resized and padded image
    ///   - The scale factor used for resizing
    ///   - The x offset used for centering
    ///   - The y offset used for centering
    /// * `Err(Box<dyn Error>)` - An error if the operation fails
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rusty_scrfd::helpers::opencv_helper::OpenCVHelper;
    /// use opencv::core::Mat;
    /// use opencv::imgcodecs;
    ///
    /// let helper = OpenCVHelper::new(127.5, 128.0);
    /// let image = imgcodecs::imread("sample_input/1.png", imgcodecs::IMREAD_COLOR)?;
    /// let (resized_image, scale) = helper.resize_with_aspect_ratio(&image, (640, 640))?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn resize_with_aspect_ratio(
        &self,
        image: &Mat,
        target_size: (i32, i32),
    ) -> Result<(Mat, f32, i32, i32), Box<dyn Error>> {
        let orig_width = image.cols() as f32;
        let orig_height = image.rows() as f32;

        let (input_width, input_height) = target_size;
        let im_ratio = orig_height / orig_width;
        let model_ratio = input_height as f32 / input_width as f32;

        // Calculate new dimensions while preserving aspect ratio
        let (new_width, new_height, x_offset, y_offset) = if im_ratio > model_ratio {
            let new_height = input_height;
            let new_width = ((input_height as f32) / im_ratio).round() as i32;
            let x_offset = (input_width - new_width) / 2;
            (new_width, new_height, x_offset, 0)
        } else {
            let new_width = input_width;
            let new_height = ((input_width as f32) * im_ratio).round() as i32;
            let y_offset = (input_height - new_height) / 2;
            (new_width, new_height, 0, y_offset)
        };

        let det_scale = new_height as f32 / orig_height;

        let mut opencv_resized_image = core::Mat::default();
        opencv::imgproc::resize(
            &image,
            &mut opencv_resized_image,
            core::Size::new(new_width, new_height),
            0.0,
            0.0,
            opencv::imgproc::INTER_LINEAR,
        )?;

        let mut det_image = core::Mat::new_rows_cols_with_default(
            input_height,
            input_width,
            core::CV_8UC3,
            core::Scalar::all(0.0),
        )?;
        let mut roi = det_image.roi_mut(core::Rect::new(x_offset, y_offset, new_width, new_height))?;
        opencv_resized_image.copy_to(&mut roi)?;

        Ok((det_image, det_scale, x_offset, y_offset))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::{Mat, Scalar};

    #[test]
    fn test_new_helper() {
        let helper = OpenCVHelper::new(127.5, 128.0);
        assert_eq!(helper.mean, 127.5);
        assert_eq!(helper.std, 128.0);
    }

    #[test]
    fn test_resize_with_aspect_ratio_landscape() -> Result<(), Box<dyn Error>> {
        // Create a test image (100x50)
        let test_image = Mat::new_rows_cols_with_default(
            50,
            100,
            core::CV_8UC3,
            Scalar::new(255.0, 0.0, 0.0, 0.0),
        )?;

        let helper = OpenCVHelper::new(127.5, 128.0);
        let (resized, scale, _, _) = helper.resize_with_aspect_ratio(&test_image, (200, 200))?;

        // Check dimensions
        assert_eq!(resized.rows(), 200);
        assert_eq!(resized.cols(), 200);

        // For landscape image (100x50):
        // im_ratio = 50/100 = 0.5
        // model_ratio = 200/200 = 1.0
        // Since im_ratio < model_ratio:
        // new_width = 200
        // new_height = (200 * 0.5) = 100
        // scale = 100/50 = 2.0
        assert!((scale - 2.0).abs() < 0.01);

        Ok(())
    }

    #[test]
    fn test_resize_with_aspect_ratio_portrait() -> Result<(), Box<dyn Error>> {
        // Create a test image (50x100)
        let test_image = Mat::new_rows_cols_with_default(
            100,
            50,
            core::CV_8UC3,
            Scalar::new(255.0, 0.0, 0.0, 0.0),
        )?;

        let helper = OpenCVHelper::new(127.5, 128.0);
        let (resized, scale, _, _) = helper.resize_with_aspect_ratio(&test_image, (200, 200))?;

        // Check dimensions
        assert_eq!(resized.rows(), 200);
        assert_eq!(resized.cols(), 200);

        // For portrait image (50x100):
        // im_ratio = 100/50 = 2.0
        // model_ratio = 200/200 = 1.0
        // Since im_ratio > model_ratio:
        // new_height = 200
        // new_width = (200/2.0) = 100
        // scale = 200/100 = 2.0
        assert!((scale - 2.0).abs() < 0.01);

        Ok(())
    }

    #[test]
    fn test_prepare_input_tensor() -> Result<(), Box<dyn Error>> {
        // Create a test image (100x100)
        let test_image = Mat::new_rows_cols_with_default(
            100,
            100,
            core::CV_8UC3,
            Scalar::new(255.0, 0.0, 0.0, 0.0),
        )?;

        let helper = OpenCVHelper::new(127.5, 128.0);
        let tensor = helper.prepare_input_tensor(&test_image, (100, 100))?;

        // Check tensor shape (batch_size, channels, height, width)
        assert_eq!(tensor.shape(), &[1, 3, 100, 100]);

        // Check that values are normalized
        // The mean should be close to 0 after normalization, but not exactly 0
        // due to floating point arithmetic and OpenCV's normalization process
        let mean = tensor.mean().unwrap();
        assert!(mean.abs() < 1.0); // Allow for some numerical error

        Ok(())
    }

    #[test]
    fn test_prepare_input_tensor_with_different_sizes() -> Result<(), Box<dyn Error>> {
        // Create a test image (100x100)
        let test_image = Mat::new_rows_cols_with_default(
            100,
            100,
            core::CV_8UC3,
            Scalar::new(255.0, 0.0, 0.0, 0.0),
        )?;

        let helper = OpenCVHelper::new(127.5, 128.0);
        let tensor = helper.prepare_input_tensor(&test_image, (200, 200))?;

        // Check tensor shape (batch_size, channels, height, width)
        assert_eq!(tensor.shape(), &[1, 3, 200, 200]);

        Ok(())
    }
}
