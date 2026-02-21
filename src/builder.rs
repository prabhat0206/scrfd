//! Builder module for SCRFD face detection models.
//!
//! This module provides a builder pattern implementation for constructing both synchronous
//! and asynchronous SCRFD face detection models with configurable parameters.
//!
//! # Example
//! ```no_run
//! use rusty_scrfd::builder::SCRFDBuilder;
//! use ort::session::Session;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let session = Session::builder()?.commit_from_file("model.onnx")?;
//!
//! // Build synchronous model with default parameters
//! let model = SCRFDBuilder::new(session)
//!     .build()?;
//!
//! // Or customize parameters
//! let session = Session::builder()?.commit_from_file("model.onnx")?;
//! let model = SCRFDBuilder::new(session)
//!     .set_input_size((320, 320))
//!     .set_conf_thres(0.6)
//!     .set_iou_thres(0.4)
//!     .build()?;
//! # Ok(())
//! # }
//! ```

use crate::scrfd::SCRFD;
use ort::session::Session;
use std::error::Error;

#[cfg(feature = "async")]
use crate::scrfd_async::SCRFDAsync;

/// Builder for configuring and constructing SCRFD face detection models
///
/// This struct provides a fluent builder interface for creating both synchronous [`SCRFD`]
/// and asynchronous [`SCRFDAsync`] model instances with customizable parameters.
///
/// The builder allows setting:
/// - Input image dimensions
/// - Confidence threshold for detections
/// - IoU (Intersection over Union) threshold for non-maximum suppression
/// - ONNX Runtime session
///
/// Default values are:
/// - Input size: (640, 640)
/// - Confidence threshold: 0.25
/// - IoU threshold: 0.4
pub struct SCRFDBuilder {
    session: Session,
    input_size: Option<(i32, i32)>,
    conf_thres: Option<f32>,
    iou_thres: Option<f32>,
    relative_output: bool,
}

impl SCRFDBuilder {
    /// Creates a new SCRFD builder with default parameters
    ///
    /// # Arguments
    /// * `session` - ONNX Runtime session for the model
    ///
    /// # Returns
    /// A new builder instance with default parameters:
    /// - Input size: (640, 640)
    /// - Confidence threshold: 0.25
    /// - IoU threshold: 0.4
    pub fn new(session: Session) -> Self {
        SCRFDBuilder {
            session,
            input_size: Some((640, 640)),
            conf_thres: Some(0.25),
            iou_thres: Some(0.4),
            relative_output: true,
        }
    }

    /// Sets the input image dimensions
    ///
    /// # Arguments
    /// * `size` - Tuple of (width, height) for the input image
    ///
    /// # Returns
    /// A mutable reference to self for method chaining
    pub fn set_input_size(mut self, size: (i32, i32)) -> Self {
        self.input_size = Some(size);
        self
    }

    /// Sets the confidence threshold for face detection
    ///
    /// # Arguments
    /// * `thres` - Confidence threshold (0.0 to 1.0)
    ///
    /// # Returns
    /// A mutable reference to self for method chaining
    pub fn set_conf_thres(mut self, thres: f32) -> Self {
        self.conf_thres = Some(thres);
        self
    }

    /// Sets the IoU threshold for non-maximum suppression
    ///
    /// # Arguments
    /// * `thres` - IoU threshold (0.0 to 1.0)
    ///
    /// # Returns
    /// A mutable reference to self for method chaining
    pub fn set_iou_thres(mut self, thres: f32) -> Self {
        self.iou_thres = Some(thres);
        self
    }

    /// Sets the relative output flag
    ///
    /// # Arguments
    /// * `relative` - Whether to use relative output
    ///
    /// # Returns
    /// A mutable reference to self for method chaining
    pub fn set_relative_output(mut self, relative: bool) -> Self {
        self.relative_output = relative;
        self
    }

    /// Builds a synchronous SCRFD model with the configured parameters
    ///
    /// # Returns
    /// A Result containing either:
    /// * `Ok(SCRFD)` - Successfully built model
    /// * `Err(Box<dyn Error>)` - Error during model construction
    pub fn build(self) -> Result<SCRFD, Box<dyn Error>> {
        let input_size = self.input_size.ok_or("Input size not set")?;
        let conf_thres = self.conf_thres.ok_or("Confidence threshold not set")?;
        let iou_thres = self.iou_thres.ok_or("IoU threshold not set")?;

        SCRFD::new(
            self.session,
            input_size,
            conf_thres,
            iou_thres,
            self.relative_output,
        )
    }

    /// Builds an asynchronous SCRFD model with the configured parameters
    ///
    /// # Returns
    /// A Result containing either:
    /// * `Ok(SCRFDAsync)` - Successfully built model
    /// * `Err(Box<dyn Error>)` - Error during model construction
    #[cfg(feature = "async")]
    pub fn build_async(self) -> Result<SCRFDAsync, Box<dyn Error>> {
        let input_size = self.input_size.ok_or("Input size not set")?;
        let conf_thres = self.conf_thres.ok_or("Confidence threshold not set")?;
        let iou_thres = self.iou_thres.ok_or("IoU threshold not set")?;

        SCRFDAsync::new(
            self.session,
            input_size,
            conf_thres,
            iou_thres,
            self.relative_output,
        )
    }
}
