//! Module for converting between absolute and relative coordinates in image processing.
//!
//! This module provides utilities for converting bounding boxes and keypoints between
//! absolute pixel coordinates and relative normalized coordinates (0-1 range).
//!
//! # Examples
//!
//! ```rust
//! use ndarray::array;
//! use rusty_scrfd::helpers::relative_conversion::RelativeConversion;
//!
//! // Convert bounding boxes (score column is preserved)
//! let bboxes = array![[100.0, 100.0, 200.0, 200.0, 0.95]];
//! let relative_boxes = RelativeConversion::absolute_to_relative_bboxes(&bboxes, 400, 400);
//!
//! // Convert keypoints
//! let keypoints = array![[[100.0, 100.0], [200.0, 200.0]]];
//! let relative_keypoints = RelativeConversion::absolute_to_relative_keypoints(&keypoints, 400, 400);
//! ```

use ndarray::{Array2, Array3, Axis};

/// A utility struct for converting between absolute and relative coordinates.
///
/// This struct provides methods to convert bounding boxes and keypoints from
/// absolute pixel coordinates to relative normalized coordinates (0-1 range).
/// The conversion is useful for:
/// - Normalizing coordinates for machine learning models
/// - Making coordinates resolution-independent
/// - Standardizing coordinate formats across different image sizes
pub struct RelativeConversion;

impl RelativeConversion {
    /// Converts absolute bounding boxes to relative coordinates.
    ///
    /// This function takes bounding boxes in the format `[x1, y1, x2, y2]` and converts
    /// them to the format `[left, top, width, height]` with values normalized between 0 and 1.
    ///
    /// # Arguments
    ///
    /// * `bboxes` - A reference to a 2D array containing bounding boxes in `[x1, y1, x2, y2]` format.
    ///              Each row represents one bounding box.
    /// * `img_width` - The width of the image in pixels.
    /// * `img_height` - The height of the image in pixels.
    ///
    /// # Returns
    ///
    /// A 2D array containing bounding boxes in `[left, top, width, height, ...]` format with
    /// coordinate values normalized between 0 and 1. Any extra columns beyond the first 4
    /// (e.g. confidence scores) are preserved as-is.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::array;
    /// use rusty_scrfd::helpers::relative_conversion::RelativeConversion;
    ///
    /// let bboxes = array![[100.0, 100.0, 200.0, 200.0, 0.95]];
    /// let relative_boxes = RelativeConversion::absolute_to_relative_bboxes(&bboxes, 400, 400);
    /// // relative_boxes will contain [[0.25, 0.25, 0.25, 0.25, 0.95]]
    /// ```
    ///
    /// # Panics
    ///
    /// This function will panic if:
    /// * The input array has incorrect dimensions
    /// * The image dimensions are zero
    pub fn absolute_to_relative_bboxes(
        bboxes: &Array2<f32>,
        img_width: u32,
        img_height: u32,
    ) -> Array2<f32> {
        // Convert image dimensions to f32 for division
        let img_width_f = img_width as f32;
        let img_height_f = img_height as f32;

        let ncols = bboxes.ncols();

        // Preserve all columns (coords + any extra such as score)
        let mut relative_bboxes = Array2::<f32>::zeros((bboxes.nrows(), ncols));

        for (i, bbox) in bboxes.axis_iter(Axis(0)).enumerate() {
            let x1 = bbox[0];
            let y1 = bbox[1];
            let x2 = bbox[2];
            let y2 = bbox[3];

            // Calculate relative coordinates
            relative_bboxes[[i, 0]] = x1 / img_width_f;
            relative_bboxes[[i, 1]] = y1 / img_height_f;
            relative_bboxes[[i, 2]] = (x2 - x1) / img_width_f;
            relative_bboxes[[i, 3]] = (y2 - y1) / img_height_f;

            // Copy any extra columns (e.g. confidence score) unchanged
            for c in 4..ncols {
                relative_bboxes[[i, c]] = bbox[c];
            }
        }

        relative_bboxes
    }

    /// Converts absolute keypoints to relative coordinates.
    ///
    /// This function takes keypoints in absolute pixel coordinates and converts them to
    /// relative coordinates normalized between 0 and 1. The function also handles edge cases
    /// by clamping values to the valid range [0, 1].
    ///
    /// # Arguments
    ///
    /// * `keypoints` - A reference to a 3D array containing keypoints in `[x, y]` format.
    ///                 Shape: `(num_detections, num_keypoints, 2)`
    /// * `img_width` - The width of the image in pixels.
    /// * `img_height` - The height of the image in pixels.
    ///
    /// # Returns
    ///
    /// A 3D array containing keypoints in `[x_rel, y_rel]` format with values normalized
    /// between 0 and 1. Values outside the valid range are clamped to [0, 1].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::array;
    /// use rusty_scrfd::helpers::relative_conversion::RelativeConversion;
    ///
    /// let keypoints = array![[[100.0, 100.0], [200.0, 200.0]]];
    /// let relative_keypoints = RelativeConversion::absolute_to_relative_keypoints(&keypoints, 400, 400);
    /// // relative_keypoints will contain [[[0.25, 0.25], [0.5, 0.5]]]
    /// ```
    ///
    /// # Panics
    ///
    /// This function will panic if:
    /// * The input array has incorrect dimensions
    /// * The image dimensions are zero
    pub fn absolute_to_relative_keypoints(
        keypoints: &Array3<f32>,
        img_width: u32,
        img_height: u32,
    ) -> Array3<f32> {
        let img_width_f = img_width as f32;
        let img_height_f = img_height as f32;

        // Initialize a new Array3 for relative keypoints
        let mut relative_keypoints = Array3::<f32>::zeros(keypoints.dim());

        for (i, kp_set) in keypoints.axis_iter(Axis(0)).enumerate() {
            for (j, kp) in kp_set.axis_iter(Axis(0)).enumerate() {
                let x_rel = kp[0] / img_width_f;
                let y_rel = kp[1] / img_height_f;

                // Clamp values between 0 and 1 to handle edge cases
                relative_keypoints[[i, j, 0]] = x_rel.clamp(0.0, 1.0);
                relative_keypoints[[i, j, 1]] = y_rel.clamp(0.0, 1.0);
            }
        }

        relative_keypoints
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_absolute_to_relative_bboxes() {
        // Test case 1: Single bounding box with score
        let bboxes = array![[100.0, 100.0, 200.0, 200.0, 0.95]];
        let result = RelativeConversion::absolute_to_relative_bboxes(&bboxes, 400, 400);
        assert_eq!(result.shape(), &[1, 5]);
        assert!((result[[0, 0]] - 0.25).abs() < 1e-6); // left
        assert!((result[[0, 1]] - 0.25).abs() < 1e-6); // top
        assert!((result[[0, 2]] - 0.25).abs() < 1e-6); // width
        assert!((result[[0, 3]] - 0.25).abs() < 1e-6); // height
        assert!((result[[0, 4]] - 0.95).abs() < 1e-6); // score preserved

        // Test case 2: Multiple bounding boxes with scores
        let bboxes = array![[0.0, 0.0, 100.0, 100.0, 0.9], [200.0, 200.0, 300.0, 300.0, 0.8]];
        let result = RelativeConversion::absolute_to_relative_bboxes(&bboxes, 400, 400);
        assert_eq!(result.shape(), &[2, 5]);

        // First bbox
        assert!((result[[0, 0]] - 0.0).abs() < 1e-6); // left
        assert!((result[[0, 1]] - 0.0).abs() < 1e-6); // top
        assert!((result[[0, 2]] - 0.25).abs() < 1e-6); // width
        assert!((result[[0, 3]] - 0.25).abs() < 1e-6); // height
        assert!((result[[0, 4]] - 0.9).abs() < 1e-6);  // score preserved

        // Second bbox
        assert!((result[[1, 0]] - 0.5).abs() < 1e-6); // left
        assert!((result[[1, 1]] - 0.5).abs() < 1e-6); // top
        assert!((result[[1, 2]] - 0.25).abs() < 1e-6); // width
        assert!((result[[1, 3]] - 0.25).abs() < 1e-6); // height
        assert!((result[[1, 4]] - 0.8).abs() < 1e-6);  // score preserved

        // Test case 3: Bbox without score (4 columns only)
        let bboxes = array![[100.0, 100.0, 200.0, 200.0]];
        let result = RelativeConversion::absolute_to_relative_bboxes(&bboxes, 400, 400);
        assert_eq!(result.shape(), &[1, 4]);
        assert!((result[[0, 0]] - 0.25).abs() < 1e-6); // left
        assert!((result[[0, 1]] - 0.25).abs() < 1e-6); // top
        assert!((result[[0, 2]] - 0.25).abs() < 1e-6); // width
        assert!((result[[0, 3]] - 0.25).abs() < 1e-6); // height
    }

    #[test]
    fn test_absolute_to_relative_keypoints() {
        // Test case 1: Single detection with multiple keypoints
        let keypoints = array![[[100.0, 100.0], [200.0, 200.0], [300.0, 300.0]]];
        let result = RelativeConversion::absolute_to_relative_keypoints(&keypoints, 400, 400);
        assert_eq!(result.shape(), &[1, 3, 2]);

        // Check first keypoint
        assert!((result[[0, 0, 0]] - 0.25).abs() < 1e-6); // x
        assert!((result[[0, 0, 1]] - 0.25).abs() < 1e-6); // y

        // Check second keypoint
        assert!((result[[0, 1, 0]] - 0.5).abs() < 1e-6); // x
        assert!((result[[0, 1, 1]] - 0.5).abs() < 1e-6); // y

        // Check third keypoint
        assert!((result[[0, 2, 0]] - 0.75).abs() < 1e-6); // x
        assert!((result[[0, 2, 1]] - 0.75).abs() < 1e-6); // y

        // Test case 2: Multiple detections with keypoints
        let keypoints = array![
            [[100.0, 100.0], [200.0, 200.0]],
            [[300.0, 300.0], [350.0, 350.0]]
        ];
        let result = RelativeConversion::absolute_to_relative_keypoints(&keypoints, 400, 400);
        assert_eq!(result.shape(), &[2, 2, 2]);

        // Check first detection
        assert!((result[[0, 0, 0]] - 0.25).abs() < 1e-6); // x
        assert!((result[[0, 0, 1]] - 0.25).abs() < 1e-6); // y
        assert!((result[[0, 1, 0]] - 0.5).abs() < 1e-6); // x
        assert!((result[[0, 1, 1]] - 0.5).abs() < 1e-6); // y

        // Check second detection
        assert!((result[[1, 0, 0]] - 0.75).abs() < 1e-6); // x
        assert!((result[[1, 0, 1]] - 0.75).abs() < 1e-6); // y
        assert!((result[[1, 1, 0]] - 0.875).abs() < 1e-6); // x
        assert!((result[[1, 1, 1]] - 0.875).abs() < 1e-6); // y

        // Test case 3: Keypoints outside image bounds (should be clamped)
        let keypoints = array![[[-100.0, -100.0], [500.0, 500.0]]];
        let result = RelativeConversion::absolute_to_relative_keypoints(&keypoints, 400, 400);
        assert_eq!(result.shape(), &[1, 2, 2]);

        // Check clamped values
        assert!((result[[0, 0, 0]] - 0.0).abs() < 1e-6); // x
        assert!((result[[0, 0, 1]] - 0.0).abs() < 1e-6); // y
        assert!((result[[0, 1, 0]] - 1.0).abs() < 1e-6); // x
        assert!((result[[0, 1, 1]] - 1.0).abs() < 1e-6); // y
    }
}
