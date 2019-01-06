//! Items related to nannou's dynamic render pass type.
//!
//! Vulkano itself provides `single_pass_renderpass!` and `ordered_passes_renderpass!` macros which
//! are useful for creating fixed-size render pass types that are checked for correctness (as much
//! as is possible) at compile time.
//!
//! While this is useful for many cases, it makes it difficult to change parameters on the
//! renderpass at runtime. For example, in order to change the load operation of a renderpass at
//! runtime when using the vulkano macros, one must create two fully independent renderpass types -
//! one with the first desired load op parameter and another with the second.
//!
//! Nannou's `render_pass::Description` type aims to simplify this by providing a more flexible,
//! dynamic `RenderPassDesc` implementation that is checked for correctness at runtime rather than
//! compile time.

use vulkano::format::{ClearValue, Format};
use vulkano::framebuffer::{AttachmentDescription, PassDependencyDescription, PassDescription,
                           RenderPassDesc, RenderPassDescClearValues};
use vulkano::image::ImageLayout;

/// A dynamic representation of a render pass description.
///
/// `vulkano` itself provides a `RenderPassDesc` trait that allows for implementing custom render
/// pass description types. While `vulkano` provides the `single_pass_renderpass!` and
/// `ordered_passes_renderpass!` macros, these generate fixed types and do not allow for changing
/// individual values at runtime.
#[derive(Debug)]
pub struct Description {
    attachment_descriptions: Vec<AttachmentDescription>,
    subpass_descriptions: Vec<PassDescription>,
    dependency_descriptions: Vec<PassDependencyDescription>,
}

/// The error returned by `validate_descriptions`.
#[derive(Debug)]
pub enum InvalidDescriptionError {
    /// Two color/depth/stencil attachments within a single subpass had differing samples.
    InvalidSamples {
        subpass_idx: usize,
        attachment_a_idx: usize,
        attachment_a_samples: u32,
        attachment_b_idx: usize,
        attachment_b_samples: u32,
    },
    /// The subpass contained an invalid index to an attachment.
    SubpassInvalidAttachmentIndex {
        subpass_idx: usize,
        invalid_attachment_idx: usize,
    },
    /// Although the same attachment was referenced in both the `color_attachments`/`depth_stencil`
    /// field and `input_attachment` fields, their `ImageLayout`s differed.
    SubpassInvalidImageLayout {
        subpass_idx: usize,
        attachment_idx: usize,
        layout: ImageLayout,
        input_layout: ImageLayout,
    },
    /// A preserve attachment was found that was contained in one of the other members.
    SubpassInvalidPreserveAttachment {
        subpass_idx: usize,
        attachment_idx: usize,
    },
    /// A resolve attachment had a number of samples specified that was greater than one.
    SubpassInvalidResolveAttachmentSamples {
        subpass_idx: usize,
        attachment_idx: usize,
        attachment_samples: u32,
    },
    /// A color attachment had a `samples` value of 1 or 0 even though a resolve attachment was
    /// included.
    SubpassInvalidColorAttachmentSamples {
        subpass_idx: usize,
        attachment_idx: usize,
        attachment_samples: u32,
    },
    /// The subpass contained one or more resolve attachments and there was a mismatch between one
    /// of the resolve and color attachment formats.
    SubpassInvalidAttachmentFormat {
        subpass_idx: usize,
        attachment_a_idx: usize,
        attachment_a_format: Format,
        attachment_b_idx: usize,
        attachment_b_format: Format,
    },
}

/// Checks the validity of each of the given description lists.
pub fn validate_descriptions(
    attachment_descriptions: &[AttachmentDescription],
    subpass_descriptions: &[PassDescription],
    dependency_descriptions: &[PassDependencyDescription],
) -> Result<(), InvalidDescriptionError> {
    // Validate subpass attachment indices.
    for (subpass_idx, subpass_desc) in subpass_descriptions.iter().enumerate() {
         let attachment_indices = subpass_desc
            .color_attachments
            .iter()
            .chain(subpass_desc.depth_stencil.as_ref())
            .chain(&subpass_desc.input_attachments)
            .map(|&(attachment_idx, _)| attachment_idx)
            .chain(subpass_desc.preserve_attachments.iter().cloned());
         for attachment_idx in attachment_indices {
             if let None = attachment_descriptions.get(attachment_idx) {
                 return Err(InvalidDescriptionError::SubpassInvalidAttachmentIndex {
                     subpass_idx,
                     invalid_attachment_idx: attachment_idx,
                 });
             }
         }
    }

    // Validate sample counts.
    for (subpass_idx, subpass_desc) in subpass_descriptions.iter().enumerate() {
        let mut attachment_indices = subpass_desc
            .color_attachments
            .iter()
            .chain(subpass_desc.depth_stencil.as_ref())
            .map(|&(attachment_idx, _)| attachment_idx);
        if let Some(first_idx) = attachment_indices.next() {
            // The number of samples in the first attachment.
            let samples = attachment_descriptions[first_idx].samples;
            for attachment_idx in attachment_indices {
                let samples_b = attachment_descriptions[attachment_idx].samples;
                if samples != samples_b {
                    return Err(InvalidDescriptionError::InvalidSamples {
                        subpass_idx,
                        attachment_a_idx: first_idx,
                        attachment_a_samples: samples,
                        attachment_b_idx: attachment_idx,
                        attachment_b_samples: samples_b,
                    });
                }
            }
        }
    }

    // Validate `ImageLayout`s.
    for (subpass_idx, subpass_desc) in subpass_descriptions.iter().enumerate() {
        for &(idx_a, layout_a) in subpass_desc
            .color_attachments
            .iter()
            .chain(subpass_desc.depth_stencil.as_ref())
        {
            for &(idx_b, layout_b) in &subpass_desc.input_attachments {
                if idx_a == idx_b && layout_a != layout_b {
                    return Err(InvalidDescriptionError::SubpassInvalidImageLayout {
                        subpass_idx,
                        attachment_idx: idx_a,
                        layout: layout_a,
                        input_layout: layout_b,
                    });
                }
            }
        }
    }

    // Validate `preserve_members`.
    for (subpass_idx, subpass_desc) in subpass_descriptions.iter().enumerate() {
        if let Some(attachment_idx) = subpass_desc
            .color_attachments
            .iter()
            .chain(subpass_desc.depth_stencil.as_ref())
            .chain(&subpass_desc.input_attachments)
            .map(|&(attachment_idx, _)| attachment_idx)
            .find(|&attachment_idx| {
                subpass_desc.preserve_attachments.iter().any(|&idx| attachment_idx == idx)
            })
        {
            return Err(InvalidDescriptionError::SubpassInvalidPreserveAttachment {
                subpass_idx,
                attachment_idx,
            });
        }
    }

    // Validate `resolve_attachments`.
    for (subpass_idx, subpass_desc) in subpass_descriptions.iter().enumerate() {
        if subpass_desc.resolve_attachments.is_empty() {
            continue;
        }

        // Check that all resolve attachments have one sample.
        for &(attachment_idx, _img_layout) in &subpass_desc.resolve_attachments {
            let attachment_samples = attachment_descriptions[attachment_idx].samples;
            if attachment_samples != 1 {
                return Err(InvalidDescriptionError::SubpassInvalidResolveAttachmentSamples {
                    subpass_idx,
                    attachment_idx,
                    attachment_samples,
                });
            }
        }

        // Check all color attachments have more than one sample.
        for &(attachment_idx, _img_layout) in &subpass_desc.color_attachments {
            let attachment_samples = attachment_descriptions[attachment_idx].samples;
            if attachment_samples <= 1 {
                return Err(InvalidDescriptionError::SubpassInvalidColorAttachmentSamples {
                    subpass_idx,
                    attachment_idx,
                    attachment_samples,
                });
            }
        }

        // Check that all resolve attachments and color attachments have the same format.
        let mut attachment_indices = subpass_desc
            .resolve_attachments
            .iter()
            .chain(&subpass_desc.color_attachments)
            .map(|&(attachment_idx, _)| attachment_idx);
        if let Some(first_idx) = attachment_indices.next() {
            let format = attachment_descriptions[first_idx].format;
            for attachment_idx in attachment_indices {
                let format_b = attachment_descriptions[attachment_idx].format;
                if format != format_b {
                    return Err(InvalidDescriptionError::SubpassInvalidAttachmentFormat {
                        subpass_idx,
                        attachment_a_idx: first_idx,
                        attachment_a_format: format,
                        attachment_b_idx: attachment_idx,
                        attachment_b_format: format_b,
                    });
                }
            }
        }
    }

    // Validate `LoadOp` of first `input_attachments`.
    if let Some((subpass_idx, subpass_desc)) = subpass_descriptions.iter().enumerate().next() {
        for &(attachment_idx, _) in &subpass_desc.input_attachments {

        }
    }

    Ok(())
}

unsafe impl RenderPassDescClearValues<Vec<ClearValue>> for Description {
    fn convert_clear_values(&self, vals: Vec<ClearValue>) -> Box<Iterator<Item = ClearValue>> {
        if self.attachment_descriptions.len() != vals.len() {
            panic!(
                "mismatch between number of attachments ({}) and number of clear values ({})",
                self.attachment_descriptions.len(),
                vals.len()
            );
        }
        Box::new(vals.into_iter()) as Box<_>
    }
}

// The `RenderPassDesc` trait is unsafe as it requires the implementor to guarantee a set of
// invariants required for vulkan interop to behave as expected. You can find these invariants
// [here](https://docs.rs/vulkano/latest/vulkano/framebuffer/trait.RenderPassDesc.html).
unsafe impl RenderPassDesc for Description {
    fn num_attachments(&self) -> usize {
        self.attachment_descriptions.len()
    }

    fn attachment_desc(&self, num: usize) -> Option<AttachmentDescription> {
        self.attachment_descriptions.get(num).map(Clone::clone)
    }

    fn num_subpasses(&self) -> usize {
        self.subpass_descriptions.len()
    }

    fn subpass_desc(&self, num: usize) -> Option<PassDescription> {
        self.subpass_descriptions.get(num).map(Clone::clone)
    }

    fn num_dependencies(&self) -> usize {
        self.dependency_descriptions.len()
    }

    fn dependency_desc(&self, num: usize) -> Option<PassDependencyDescription> {
        self.dependency_descriptions.get(num).map(Clone::clone)
    }
}
