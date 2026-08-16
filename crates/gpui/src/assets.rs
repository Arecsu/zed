use crate::{DevicePixels, Pixels, Result, SharedString, Size, size};
use smallvec::SmallVec;

use image::{Delay, Frame};
use std::{
    any::Any,
    borrow::Cow,
    fmt,
    hash::Hash,
    sync::Arc,
    sync::atomic::{AtomicUsize, Ordering::SeqCst},
};

/// A source of assets for this app to use.
pub trait AssetSource: 'static + Send + Sync {
    /// Load the given asset from the source path.
    fn load(&self, path: &str) -> Result<Option<Cow<'static, [u8]>>>;

    /// List the assets at the given path.
    fn list(&self, path: &str) -> Result<Vec<SharedString>>;
}

impl AssetSource for () {
    fn load(&self, _path: &str) -> Result<Option<Cow<'static, [u8]>>> {
        Ok(None)
    }

    fn list(&self, _path: &str) -> Result<Vec<SharedString>> {
        Ok(vec![])
    }
}

/// A unique identifier for the image cache
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ImageId(pub usize);

#[derive(PartialEq, Eq, Hash, Clone)]
#[expect(missing_docs)]
pub struct RenderImageParams {
    pub image_id: ImageId,
    pub frame_index: usize,
}

/// A cached and processed image, in BGRA format.
///
/// Images are normally CPU byte-backed (`image::Frame`) and uploaded into a
/// GPU texture atlas on demand. A zero-copy variant backed by an already
/// on-GPU texture is also possible via [`RenderImage::from_gpu_image`], which
/// the platform renderer samples directly (no CPU round-trip).
pub struct RenderImage {
    /// The ID associated with this image
    pub id: ImageId,
    /// The scale factor of this image on render.
    pub(crate) scale_factor: f32,
    data: SmallVec<[Frame; 1]>,
    /// A GPU-resident backing texture, when present. Mutually exclusive with
    /// `data`: a zero-copy image has no CPU bytes.
    gpu: Option<Arc<dyn GpuImage>>,
}

/// A GPU-resident image drawn by the renderer without a CPU round-trip.
///
/// Platform renderers downcast [`GpuImage::as_any`] to their native texture
/// type (e.g. a `wgpu::TextureView`) and sample it directly, skipping the
/// host-visible atlas upload that CPU byte-backed images perform.
pub trait GpuImage: Send + Sync {
    /// Size of the image in device pixels.
    fn size(&self) -> Size<DevicePixels>;
    /// Opaque handle to the platform-native GPU texture backing this image.
    fn as_any(&self) -> &dyn Any;
}

impl PartialEq for RenderImage {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for RenderImage {}

impl RenderImage {
    /// Create a new image from the given data.
    pub fn new(data: impl Into<SmallVec<[Frame; 1]>>) -> Self {
        static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

        Self {
            id: ImageId(NEXT_ID.fetch_add(1, SeqCst)),
            scale_factor: 1.0,
            data: data.into(),
            gpu: None,
        }
    }

    /// Create a zero-copy image backed by an already-uploaded GPU texture.
    ///
    /// The platform renderer samples `image`'s native texture directly,
    /// avoiding the CPU readback + atlas re-upload that [`RenderImage::new`]
    /// performs. The texture must live on the renderer's GPU device and be
    /// sampleable (e.g. a `wgpu::TextureView` with `TEXTURE_BINDING` usage).
    pub fn from_gpu_image(image: Arc<dyn GpuImage>) -> Self {
        static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

        Self {
            id: ImageId(NEXT_ID.fetch_add(1, SeqCst)),
            scale_factor: 1.0,
            data: SmallVec::new(),
            gpu: Some(image),
        }
    }

    /// Whether this image is GPU-resident (zero-copy) rather than CPU
    /// byte-backed. When `Some`, the platform renderer draws it directly.
    pub fn gpu_image(&self) -> Option<&Arc<dyn GpuImage>> {
        self.gpu.as_ref()
    }

    /// Convert this image into a byte slice.
    pub fn as_bytes(&self, frame_index: usize) -> Option<&[u8]> {
        self.data
            .get(frame_index)
            .map(|frame| frame.buffer().as_raw().as_slice())
    }

    /// Get the size of this image, in pixels.
    pub fn size(&self, frame_index: usize) -> Size<DevicePixels> {
        if let Some(gpu) = &self.gpu {
            return gpu.size();
        }
        self.data
            .get(frame_index)
            .map(|frame| {
                let (width, height) = frame.buffer().dimensions();
                size(width.into(), height.into())
            })
            .unwrap_or_default()
    }

    /// Get the size of this image, in pixels for display, adjusted for the scale factor.
    pub(crate) fn render_size(&self, frame_index: usize) -> Size<Pixels> {
        self.size(frame_index)
            .map(|v| (v.0 as f32 / self.scale_factor).into())
    }

    /// Get the delay of this frame from the previous
    pub fn delay(&self, frame_index: usize) -> Delay {
        self.data
            .get(frame_index)
            .map(|frame| frame.delay())
            .unwrap_or(Delay::from_numer_denom_ms(100, 1))
    }

    /// Get the number of frames for this image.
    pub fn frame_count(&self) -> usize {
        // A zero-copy (GPU-resident) image has no CPU `data` frames — treat it as
        // a single frame so img() doesn't skip painting it (img() bails when
        // frame_count() == 0).
        if self.gpu.is_some() {
            return 1;
        }
        self.data.len()
    }
}

impl fmt::Debug for RenderImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImageData")
            .field("id", &self.id)
            .field("size", &self.data.first().map(|f| f.buffer().dimensions()))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallvec::SmallVec;

    #[test]
    fn empty_render_image_does_not_panic() {
        let image = RenderImage::new(SmallVec::new());
        assert_eq!(image.frame_count(), 0);
        assert_eq!(image.size(0), Size::default());
        assert_eq!(image.as_bytes(0), None);
        assert_eq!(image.render_size(0), Size::default());
        assert_eq!(image.delay(0), Delay::from_numer_denom_ms(100, 1));
        let _ = format!("{image:?}");
    }
}
