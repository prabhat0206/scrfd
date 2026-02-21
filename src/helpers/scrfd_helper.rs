//! Module for SCRFD (Single-stage Center Real-time Face Detector) helper functions.
//!
//! This module provides utility functions for SCRFD face detection, including:
//! - Bounding box and keypoint decoding
//! - Non-Maximum Suppression (NMS)
//! - Anchor center generation
//! - Array concatenation utilities
//!
//! # Examples
//!
//! ```rust
//! use ndarray::array;
//! use rusty_scrfd::helpers::scrfd_helper::ScrfdHelpers;
//!
//! // Decode bounding boxes
//! let points = array![[100.0, 100.0]];
//! let distance = array![[10.0, 10.0, 20.0, 20.0]];
//! let bboxes = ScrfdHelpers::distance2bbox(&points, &distance, Some((400, 400)));
//!
//! // Perform NMS
//! let detections = array![[10.0, 10.0, 20.0, 20.0, 0.9]];
//! let keep_indices = ScrfdHelpers::nms(&detections, 0.5);
//! ```

use ndarray::{Array2, Array3, Axis};
use std::cmp::Ordering;
use std::error::Error;

/// A utility struct for SCRFD face detection helper functions.
///
/// This struct provides methods for various operations needed in SCRFD face detection:
/// - Converting distance predictions to bounding boxes and keypoints
/// - Performing Non-Maximum Suppression
/// - Generating anchor centers
/// - Array manipulation utilities
pub struct ScrfdHelpers;

impl ScrfdHelpers {
    /// Decodes distance predictions to bounding boxes.
    ///
    /// This function converts center points and their distances to the four boundaries
    /// into absolute bounding box coordinates. Optionally clamps the coordinates to
    /// the image boundaries if max_shape is provided.
    ///
    /// # Arguments
    ///
    /// * `points` - Center points with shape (n, 2), where each row is [x, y].
    /// * `distance` - Distances from points to boundaries with shape (n, 4),
    ///               where each row is [left, top, right, bottom].
    /// * `max_shape` - Optional tuple of (height, width) to clamp coordinates.
    ///
    /// # Returns
    ///
    /// * `Array2<f32>` - Decoded bounding boxes with shape (n, 4),
    ///                  where each row is [x1, y1, x2, y2].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::array;
    /// use rusty_scrfd::helpers::scrfd_helper::ScrfdHelpers;
    ///
    /// let points = array![[100.0, 100.0]];
    /// let distance = array![[10.0, 10.0, 20.0, 20.0]];
    /// let bboxes = ScrfdHelpers::distance2bbox(&points, &distance, Some((400, 400)));
    /// // bboxes will contain [[90.0, 90.0, 120.0, 120.0]]
    /// ```
    pub fn distance2bbox(
        points: &Array2<f32>,
        distance: &Array2<f32>,
        max_shape: Option<(usize, usize)>,
    ) -> Array2<f32> {
        let mut x1 = &points.column(0) - &distance.column(0);
        let mut y1 = &points.column(1) - &distance.column(1);
        let mut x2 = &points.column(0) + &distance.column(2);
        let mut y2 = &points.column(1) + &distance.column(3);

        // Optionally clamp the values if max_shape is provided
        let (x1, y1, x2, y2) = if let Some((height, width)) = max_shape {
            let width = width as f32;
            let height = height as f32;
            x1.mapv_inplace(|x| x.max(0.0).min(width));
            y1.mapv_inplace(|y| y.max(0.0).min(height));
            x2.mapv_inplace(|x| x.max(0.0).min(width));
            y2.mapv_inplace(|y| y.max(0.0).min(height));
            (x1, y1, x2, y2)
        } else {
            // Do not clamp if max_shape is None
            (x1, y1, x2, y2)
        };

        let concatenated =
            ndarray::stack(Axis(1), &[x1.view(), y1.view(), x2.view(), y2.view()]).unwrap();
        concatenated
    }

    /// Decodes distance predictions to keypoints.
    ///
    /// This function converts center points and their distances to keypoints
    /// into absolute keypoint coordinates. Optionally clamps the coordinates to
    /// the image boundaries if max_shape is provided.
    ///
    /// # Arguments
    ///
    /// * `points` - Center points with shape (n, 2), where each row is [x, y].
    /// * `distance` - Distances from points to keypoints with shape (n, 2k),
    ///               where k is the number of keypoints.
    /// * `max_shape` - Optional tuple of (height, width) to clamp coordinates.
    ///
    /// # Returns
    ///
    /// * `Array2<f32>` - Decoded keypoints with shape (n, 2k),
    ///                  where k is the number of keypoints.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::array;
    /// use rusty_scrfd::helpers::scrfd_helper::ScrfdHelpers;
    ///
    /// let points = array![[100.0, 100.0]];
    /// let distance = array![[10.0, 10.0, 20.0, 20.0]];
    /// let keypoints = ScrfdHelpers::distance2kps(&points, &distance, Some((400, 400)));
    /// // keypoints will contain [[110.0, 110.0, 120.0, 120.0]]
    /// ```
    pub fn distance2kps(
        points: &Array2<f32>,
        distance: &Array2<f32>,
        max_shape: Option<(usize, usize)>,
    ) -> Array2<f32> {
        let num_keypoints = distance.shape()[1] / 2;
        let mut preds = Vec::with_capacity(2 * num_keypoints);

        for i in 0..num_keypoints {
            let mut px = &points.column(0) + &distance.column(2 * i);
            let mut py = &points.column(1) + &distance.column(2 * i + 1);
            let (px, py) = if let Some((height, width)) = max_shape {
                let width = width as f32;
                let height = height as f32;
                px.mapv_inplace(|x| x.max(0.0).min(width));
                py.mapv_inplace(|y| y.max(0.0).min(height));
                (px, py)
            } else {
                (px, py)
            };
            preds.push(px.insert_axis(Axis(1)));
            preds.push(py.insert_axis(Axis(1)));
        }

        // Concatenate along Axis(1) to get an array of shape (n, 2k)
        ndarray::concatenate(Axis(1), &preds.iter().map(|a| a.view()).collect::<Vec<_>>()).unwrap()
    }

    /// Performs Non-Maximum Suppression (NMS) on detection results.
    ///
    /// This function filters out overlapping bounding boxes by keeping only the
    /// highest scoring box when the IoU (Intersection over Union) exceeds the threshold.
    ///
    /// # Arguments
    ///
    /// * `dets` - Detection results with shape (n, 5), where each row is [x1, y1, x2, y2, score].
    /// * `iou_thres` - IoU threshold for suppression (0.0 to 1.0).
    ///
    /// # Returns
    ///
    /// * `Vec<usize>` - Indices of the boxes to keep after NMS.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::array;
    /// use rusty_scrfd::helpers::scrfd_helper::ScrfdHelpers;
    ///
    /// let dets = array![
    ///     [10.0, 10.0, 20.0, 20.0, 0.9],
    ///     [12.0, 12.0, 22.0, 22.0, 0.8]
    /// ];
    /// let keep = ScrfdHelpers::nms(&dets, 0.5);
    /// // keep will contain [0] if the boxes overlap significantly
    /// ```
    pub fn nms(dets: &Array2<f32>, iou_thres: f32) -> Vec<usize> {
        if dets.is_empty() {
            return Vec::new();
        }
        let x1 = dets.column(0);
        let y1 = dets.column(1);
        let x2 = dets.column(2);
        let y2 = dets.column(3);
        let scores = dets.column(4);

        let areas = (&x2 - &x1 + 1.0) * (&y2 - &y1 + 1.0);
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_unstable_by(|&i, &j| {
            scores[j].partial_cmp(&scores[i]).unwrap_or(Ordering::Equal)
        });

        let mut keep = Vec::new();
        while !order.is_empty() {
            let i = order[0];
            keep.push(i);

            if order.len() == 1 {
                break;
            }

            let order_rest = &order[1..];

            // Extract scalar values
            let x1_i = x1[i];
            let y1_i = y1[i];
            let x2_i = x2[i];
            let y2_i = y2[i];
            let area_i = areas[i];

            // Select the rest of the array
            let mut x1_order = x1.select(Axis(0), order_rest);
            let mut y1_order = y1.select(Axis(0), order_rest);
            let mut x2_order = x2.select(Axis(0), order_rest);
            let mut y2_order = y2.select(Axis(0), order_rest);
            let areas_order = areas.select(Axis(0), order_rest);

            // Compute the coordinates of the intersection
            x1_order.mapv_inplace(|x| x1_i.max(x));
            y1_order.mapv_inplace(|y| y1_i.max(y));
            x2_order.mapv_inplace(|x| x2_i.min(x));
            y2_order.mapv_inplace(|y| y2_i.min(y));
            let (xx1, yy1, xx2, yy2) = (x1_order, y1_order, x2_order, y2_order);

            // Compute the width and height of the intersection
            let mut w = &xx2 - &xx1 + 1.0;
            w.mapv_inplace(|x| x.max(0.0));
            let mut h = &yy2 - &yy1 + 1.0;
            h.mapv_inplace(|y| y.max(0.0));
            let inter = &w * &h;
            let ovr = &inter / (area_i + &areas_order - &inter);

            // Get indices where IoU <= threshold
            let inds: Vec<usize> = ovr
                .iter()
                .enumerate()
                .filter(|&(_, &ov)| ov <= iou_thres)
                .map(|(idx, _)| idx)
                .collect();

            // Update order
            let mut new_order = Vec::with_capacity(inds.len());
            for &idx in &inds {
                new_order.push(order[idx + 1]); // +1 because we skipped order[0]
            }
            order = new_order;
        }
        keep
    }

    /// Generates anchor centers for a feature map.
    ///
    /// This function creates a grid of anchor centers based on the feature map
    /// dimensions and stride. Each center can have multiple anchors.
    ///
    /// # Arguments
    ///
    /// * `num_anchors` - Number of anchors per location.
    /// * `height` - Height of the feature map.
    /// * `width` - Width of the feature map.
    /// * `stride` - Stride of the feature map.
    ///
    /// # Returns
    ///
    /// * `Array2<f32>` - Anchor centers with shape (height * width * num_anchors, 2),
    ///                  where each row is [x, y].
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rusty_scrfd::helpers::scrfd_helper::ScrfdHelpers;
    ///
    /// let centers = ScrfdHelpers::generate_anchor_centers(2, 2, 2, 16.0);
    /// // centers will contain 8 points (2x2 grid with 2 anchors each)
    /// ```
    pub fn generate_anchor_centers(
        num_anchors: usize,
        height: usize,
        width: usize,
        stride: f32,
    ) -> Array2<f32> {
        // Create anchor centers using [x, y] ordering
        let mut anchor_centers = Array2::zeros((height * width, 2));

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                anchor_centers[[idx, 0]] = x as f32; // Assign x first
                anchor_centers[[idx, 1]] = y as f32; // Assign y second
            }
        }

        // Multiply by stride
        anchor_centers.mapv_inplace(|x| x * stride);

        // Handle multiple anchors if needed
        let anchor_centers = if num_anchors > 1 {
            let mut repeated_anchors = Array2::zeros((height * width * num_anchors, 2));

            // Repeat each point num_anchors times
            for (i, row) in anchor_centers.rows().into_iter().enumerate() {
                for j in 0..num_anchors {
                    repeated_anchors
                        .slice_mut(ndarray::s![i * num_anchors + j, ..])
                        .assign(&row);
                }
            }

            repeated_anchors
        } else {
            anchor_centers
        };

        anchor_centers
    }

    /// Concatenates multiple 2D arrays along the first axis.
    ///
    /// This function combines multiple Array2<f32> into a single array by stacking
    /// them vertically. Returns an empty array with correct dimensions if input is empty.
    ///
    /// # Arguments
    ///
    /// * `arrays` - Vector of Array2<f32> to concatenate.
    ///
    /// # Returns
    ///
    /// * `Result<Array2<f32>, Box<dyn Error>>` - Concatenated array or error.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::array;
    /// use rusty_scrfd::helpers::scrfd_helper::ScrfdHelpers;
    ///
    /// let arrays = vec![
    ///     array![[1.0, 2.0], [3.0, 4.0]],
    ///     array![[5.0, 6.0], [7.0, 8.0]]
    /// ];
    /// let result = ScrfdHelpers::concatenate_array2(&arrays).unwrap();
    /// // result will be a 4x2 array
    /// ```
    pub fn concatenate_array2(arrays: &[Array2<f32>]) -> Result<Array2<f32>, Box<dyn Error>> {
        if arrays.is_empty() {
            return Ok(Array2::<f32>::zeros((0, 0)));
        }
        Ok(ndarray::concatenate(
            Axis(0),
            &arrays.iter().map(|a| a.view()).collect::<Vec<_>>(),
        )?)
    }

    /// Concatenates multiple 3D arrays along the first axis.
    ///
    /// This function combines multiple Array3<f32> into a single array by stacking
    /// them vertically. Returns an empty array with correct dimensions if input is empty.
    ///
    /// # Arguments
    ///
    /// * `arrays` - Vector of Array3<f32> to concatenate.
    ///
    /// # Returns
    ///
    /// * `Result<Array3<f32>, Box<dyn Error>>` - Concatenated array or error.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use ndarray::array;
    /// use rusty_scrfd::helpers::scrfd_helper::ScrfdHelpers;
    ///
    /// let arrays = vec![
    ///     array![[[1.0, 2.0], [3.0, 4.0]]],
    ///     array![[[5.0, 6.0], [7.0, 8.0]]]
    /// ];
    /// let result = ScrfdHelpers::concatenate_array3(&arrays).unwrap();
    /// // result will be a 2x2x2 array
    /// ```
    pub fn concatenate_array3(arrays: &[Array3<f32>]) -> Result<Array3<f32>, Box<dyn Error>> {
        if arrays.is_empty() {
            return Ok(Array3::<f32>::zeros((0, 0, 0)));
        }
        Ok(ndarray::concatenate(
            Axis(0),
            &arrays.iter().map(|a| a.view()).collect::<Vec<_>>(),
        )?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_nms() {
        let dets = array![
            [10.0, 10.0, 20.0, 20.0, 0.9],
            [12.0, 12.0, 22.0, 22.0, 0.8],
            [15.0, 15.0, 25.0, 25.0, 0.7],
            [30.0, 30.0, 40.0, 40.0, 0.6],
        ];
        let iou_thres = 0.5;
        let keep = ScrfdHelpers::nms(&dets, iou_thres);
        assert_eq!(keep, vec![0, 2, 3]); // Updated expected result
    }

    #[test]
    fn test_distance2bbox() {
        let points = array![[10.0, 10.0], [20.0, 20.0]];
        let distance = array![[1.0, 1.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0]];
        let expected = array![[9.0, 9.0, 12.0, 12.0], [17.0, 17.0, 24.0, 24.0]];
        let bbox = ScrfdHelpers::distance2bbox(&points, &distance, Some((30, 30)));
        assert_eq!(bbox, expected);
    }

    #[test]
    fn test_distance2kps() {
        let points = array![[10.0, 10.0], [20.0, 20.0]];
        let distance = array![
            [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0],
            [6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0, 9.0, 10.0, 10.0]
        ];
        let expected = array![
            [11.0, 11.0, 12.0, 12.0, 13.0, 13.0, 14.0, 14.0, 15.0, 15.0],
            [26.0, 26.0, 27.0, 27.0, 28.0, 28.0, 29.0, 29.0, 30.0, 30.0]
        ];
        let kps = ScrfdHelpers::distance2kps(&points, &distance, Some((30, 30)));
        assert_eq!(kps, expected);
    }

    #[test]
    fn test_distance2bbox_stride_1() {
        let anchor_centers = array![[100.0, 100.0]];
        let bbox_preds = array![[0.1, 0.1, 0.2, 0.2]];
        let expected_bboxes = array![[99.9, 99.9, 100.2, 100.2]];

        let bboxes = ScrfdHelpers::distance2bbox(&anchor_centers, &bbox_preds, None);
        assert_eq!(bboxes, expected_bboxes);
    }

    #[test]
    fn test_generate_anchor_centers_multiple_anchors() {
        let height = 2;
        let width = 2;
        let stride = 16.0;
        let num_anchors = 2;

        let anchor_centers =
            ScrfdHelpers::generate_anchor_centers(num_anchors, height, width, stride);

        let expected = array![
            [0.0, 0.0],
            [0.0, 0.0],
            [16.0, 0.0],
            [16.0, 0.0],
            [0.0, 16.0],
            [0.0, 16.0],
            [16.0, 16.0],
            [16.0, 16.0]
        ];

        println!("Anchor centers: {:?}", anchor_centers);
        println!("Expected: {:?}", expected);

        assert_eq!(anchor_centers, expected);
    }
}
