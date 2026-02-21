use crate::helpers::{
    opencv_helper::OpenCVHelper, relative_conversion::RelativeConversion,
    scrfd_helper::ScrfdHelpers,
};
use ndarray::{s, Array2, Array3, ArrayD, ArrayViewD, Axis};
use opencv::core::Mat;
use opencv::prelude::MatTraitConst;
use ort::{session::Session, value::Value};
use std::{collections::HashMap, error::Error};

pub struct SCRFD {
    input_size: (i32, i32),
    conf_thres: f32,
    iou_thres: f32,
    _fmc: usize,
    feat_stride_fpn: Vec<i32>,
    num_anchors: usize,
    use_kps: bool,
    opencv_helper: OpenCVHelper,
    session: Session,
    input_names: Vec<String>,
    relative_output: bool,
}

impl SCRFD {
    /// Constructor to initialize the SCRFD model
    /// # Arguments:
    /// - session: ONNX Runtime session for the model
    /// - input_size: Tuple of (width, height) for the input image
    /// - conf_thres: Confidence threshold
    /// - iou_thres: IoU threshold
    /// # Returns:
    /// - Self
    pub fn new(
        session: Session,
        input_size: (i32, i32),
        conf_thres: f32,
        iou_thres: f32,
        relative_output: bool,
    ) -> Result<Self, Box<dyn Error>> {
        // SCRFD model parameters
        let fmc = 3;
        let feat_stride_fpn = vec![8, 16, 32];
        let num_anchors = 2;
        let use_kps = true;

        let mean = 127.5;
        let std = 128.0;

        // Get model input names
        let input_names = session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();

        Ok(SCRFD {
            input_size,
            conf_thres,
            iou_thres,
            _fmc: fmc,
            feat_stride_fpn,
            num_anchors,
            use_kps,
            opencv_helper: OpenCVHelper::new(mean, std),
            session,
            input_names,
            relative_output,
        })
    }

    /// The forward method processes the image and runs the model
    /// # Arguments:
    /// - input_tensor: &ArrayD<f32>
    /// # Returns:
    /// - Result<(Vec<Array2<f32>>, Vec<Array2<f32>>, Vec<Array3<f32>>), Box<dyn Error>>
    pub fn forward(
        &mut self,
        input_tensor: ArrayD<f32>,
        center_cache: &mut HashMap<(i32, i32, i32), Array2<f32>>,
    ) -> Result<(Vec<Array2<f32>>, Vec<Array2<f32>>, Vec<Array3<f32>>), Box<dyn Error>> {
        let mut scores_list = Vec::new();
        let mut bboxes_list = Vec::new();
        let mut kpss_list = Vec::new();
        let input_height = input_tensor.shape()[2];
        let input_width = input_tensor.shape()[3];
        let input_value = Value::from_array(input_tensor)?;
        let input_name = self.input_names[0].clone();
        let input = ort::inputs![input_name => input_value];
        // Run the model
        let session_output = match self.session.run(input) {
            Ok(output) => output,
            Err(e) => return Err(Box::new(e)),
        };

        let mut outputs = vec![];
        for (_, output) in session_output.iter().enumerate() {
            let f32_array: ArrayViewD<f32> = output.1.try_extract_array()?;
            outputs.push(f32_array.to_owned());
        }
        drop(session_output);

        let fmc = self._fmc;
        for (idx, &stride) in self.feat_stride_fpn.iter().enumerate() {
            let scores = &outputs[idx];
            let bbox_preds = outputs[idx + fmc].to_shape((outputs[idx + fmc].len() / 4, 4))?;
            let bbox_preds = (bbox_preds * stride as f32).into_owned();
            let kps_preds = (outputs[idx + fmc * 2]
                .to_shape((outputs[idx + fmc * 2].len() / 10, 10))?
                * stride as f32)
                .into_owned();

            // Determine feature map dimensions
            let height = input_height / stride as usize;
            let width = input_width / stride as usize;

            // Generate anchor centers
            let key = (height as i32, width as i32, stride);
            let anchor_centers = center_cache.entry(key).or_insert_with(|| {
                ScrfdHelpers::generate_anchor_centers(
                    self.num_anchors,
                    height,
                    width,
                    stride as f32,
                )
            });

            // Filter scores by threshold
            let pos_inds: Vec<usize> = scores
                .iter()
                .enumerate()
                .filter(|(_, &s)| s > self.conf_thres)
                .map(|(i, _)| i)
                .collect();

            if pos_inds.is_empty() {
                continue;
            }

            let pos_scores = scores.select(Axis(0), &pos_inds);
            let bboxes = ScrfdHelpers::distance2bbox(anchor_centers, &bbox_preds, None);
            let pos_bboxes = bboxes.select(Axis(0), &pos_inds);

            scores_list.push(pos_scores.to_shape((pos_scores.len(), 1))?.to_owned());
            bboxes_list.push(pos_bboxes);

            if self.use_kps {
                let kpss = ScrfdHelpers::distance2kps(anchor_centers, &kps_preds, None);
                let kpss = kpss.to_shape((kpss.shape()[0], kpss.shape()[1] / 2, 2))?;
                let pos_kpss = kpss.select(Axis(0), &pos_inds);
                kpss_list.push(pos_kpss);
            }
        }

        Ok((scores_list, bboxes_list, kpss_list))
    }

    /// Detect faces in the image
    /// # Arguments:
    /// - image: &Mat
    /// - max_num: usize
    /// - metric: &str
    /// # Returns:
    /// - Result<(Array2<f32>, Option<Array3<f32>>), Box<dyn Error>>
    pub fn detect(
        &mut self,
        image: &Mat,
        max_num: usize,
        metric: &str,
        center_cache: &mut HashMap<(i32, i32, i32), Array2<f32>>,
    ) -> Result<(Array2<f32>, Option<Array3<f32>>), Box<dyn Error>> {
        let orig_width = image.cols() as f32;
        let orig_height = image.rows() as f32;

        let (det_image, det_scale, x_offset, y_offset) = self
            .opencv_helper
            .resize_with_aspect_ratio(image, self.input_size)?;
        let input_tensor = self
            .opencv_helper
            .prepare_input_tensor(&det_image, self.input_size)?;
        let (scores_list, bboxes_list, kpss_list) =
            self.forward(input_tensor.into_dyn(), center_cache)?;

        if scores_list.is_empty() {
            return Err("No faces detected".into());
        }

        // Concatenate scores and bboxes, then remap from canvas coords to
        // original image coords: original = (canvas - offset) / det_scale
        let scores = ScrfdHelpers::concatenate_array2(&scores_list)?;
        let mut bboxes = ScrfdHelpers::concatenate_array2(&bboxes_list)?;
        let x_off = x_offset as f32;
        let y_off = y_offset as f32;
        for mut row in bboxes.rows_mut() {
            row[0] = (row[0] - x_off) / det_scale; // x1
            row[1] = (row[1] - y_off) / det_scale; // y1
            row[2] = (row[2] - x_off) / det_scale; // x2
            row[3] = (row[3] - y_off) / det_scale; // y2
        }

        let mut kpss = if self.use_kps {
            let mut kpss = ScrfdHelpers::concatenate_array3(&kpss_list)?;
            for mut face in kpss.outer_iter_mut() {
                for mut kp in face.rows_mut() {
                    kp[0] = (kp[0] - x_off) / det_scale; // x
                    kp[1] = (kp[1] - y_off) / det_scale; // y
                }
            }
            Some(kpss)
        } else {
            None
        };

        let scores_ravel = scores.iter().collect::<Vec<_>>();
        let mut order = (0..scores_ravel.len()).collect::<Vec<usize>>();
        order.sort_unstable_by(|&i, &j| {
            scores_ravel[j]
                .partial_cmp(&scores_ravel[i])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Prepare pre_detections
        let mut pre_det = ndarray::concatenate(Axis(1), &[bboxes.view(), scores.view()])?;
        pre_det = pre_det.select(Axis(0), &order);

        let keep = ScrfdHelpers::nms(&pre_det, self.iou_thres);
        let det = pre_det.select(Axis(0), &keep);

        if self.use_kps {
            if let Some(ref mut kpss_array) = kpss {
                *kpss_array = kpss_array.select(Axis(0), &order);
                *kpss_array = kpss_array.select(Axis(0), &keep);
            }
        }

        let det = if max_num > 0 && max_num < det.shape()[0] {
            let area = (&det.slice(s![.., 2]) - &det.slice(s![.., 0]))
                * (&det.slice(s![.., 3]) - &det.slice(s![.., 1]));
            let image_center = (
                orig_width / 2.0,
                orig_height / 2.0,
            );
            let offsets = ndarray::stack![
                Axis(0),
                (&det.slice(s![.., 0]) + &det.slice(s![.., 2])) / 2.0 - image_center.1 as f32,
                (&det.slice(s![.., 1]) + &det.slice(s![.., 3])) / 2.0 - image_center.0 as f32,
            ];
            let offset_dist_squared = offsets.mapv(|x| x * x).sum_axis(Axis(0));
            let values = if metric == "max" {
                area.to_owned()
            } else {
                &area - &(offset_dist_squared * 2.0)
            };
            let mut bindex = (0..values.len()).collect::<Vec<usize>>();
            bindex.sort_unstable_by(|&i, &j| {
                values[j]
                    .partial_cmp(&values[i])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            bindex.truncate(max_num);
            let det = det.select(Axis(0), &bindex);
            if self.use_kps {
                if let Some(ref mut kpss_array) = kpss {
                    *kpss_array = kpss_array.select(Axis(0), &bindex);
                }
            }
            det
        } else {
            det
        };

        let bounding_boxes = if self.relative_output {
            RelativeConversion::absolute_to_relative_bboxes(
                &det,
                orig_width as u32,
                orig_height as u32,
            )
        } else {
            det
        };

        let keypoints = if let Some(kpss) = kpss {
            if self.relative_output {
                Some(RelativeConversion::absolute_to_relative_keypoints(
                    &kpss,
                    orig_width as u32,
                    orig_height as u32,
                ))
            } else {
                Some(kpss)
            }
        } else {
            None
        };

        Ok((bounding_boxes, keypoints))
    }
}
