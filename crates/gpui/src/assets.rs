use crate::{DevicePixels, Pixels, Result, SharedString, Size, size};
use smallvec::SmallVec;

use image::{Delay, Frame};
use std::{
    any::Any,
    borrow::Cow,
    fmt,
    hash::Hash,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering::SeqCst},
    },
};

static NEXT_IMAGE_ID: AtomicUsize = AtomicUsize::new(0);

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

/// An immutable native image supplied by a platform renderer.
///
/// The handle is intentionally opaque to GPUI. A platform atlas may downcast
/// it to its own image type, while the scene and element layers retain only
/// the stable image identity and dimensions. Native images have no CPU frame
/// bytes, so they must never enter the ordinary byte-upload callback.
#[derive(Clone)]
pub struct ExternalImage {
    size: Size<DevicePixels>,
    handle: Arc<dyn Any + Send + Sync>,
}

impl ExternalImage {
    /// Create an opaque platform-owned image descriptor.
    pub fn new(size: Size<DevicePixels>, handle: Arc<dyn Any + Send + Sync>) -> Self {
        Self { size, handle }
    }

    /// Return the physical dimensions of the native image.
    pub fn size(&self) -> Size<DevicePixels> {
        self.size
    }

    /// Return the opaque native handle for the active platform atlas.
    pub fn handle(&self) -> &(dyn Any + Send + Sync) {
        self.handle.as_ref()
    }
}

/// A cached and processed image, in BGRA format, or an immutable native image.
pub struct RenderImage {
    /// The ID associated with this image
    pub id: ImageId,
    /// The scale factor of this image on render.
    pub(crate) scale_factor: f32,
    data: SmallVec<[Frame; 1]>,
    external: Option<ExternalImage>,
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
        Self {
            id: ImageId(NEXT_IMAGE_ID.fetch_add(1, SeqCst)),
            scale_factor: 1.0,
            data: data.into(),
            external: None,
        }
    }

    /// Create an image whose pixels are owned by a platform renderer.
    pub fn new_external(size: Size<DevicePixels>, handle: Arc<dyn Any + Send + Sync>) -> Self {
        Self {
            id: ImageId(NEXT_IMAGE_ID.fetch_add(1, SeqCst)),
            scale_factor: 1.0,
            data: SmallVec::new(),
            external: Some(ExternalImage::new(size, handle)),
        }
    }

    /// Return the native image descriptor, if this image is platform-owned.
    pub fn external(&self) -> Option<&ExternalImage> {
        self.external.as_ref()
    }

    /// Convert this image into a byte slice.
    pub fn as_bytes(&self, frame_index: usize) -> Option<&[u8]> {
        self.data
            .get(frame_index)
            .map(|frame| frame.buffer().as_raw().as_slice())
    }

    /// Get the size of this image, in pixels.
    pub fn size(&self, frame_index: usize) -> Size<DevicePixels> {
        if let Some(external) = &self.external {
            return (frame_index == 0)
                .then_some(external.size)
                .unwrap_or_default();
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
        if self.external.is_some() {
            1
        } else {
            self.data.len()
        }
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

    #[test]
    fn external_render_image_has_one_frame_without_cpu_bytes() {
        let image = RenderImage::new_external(size(8u32.into(), 6u32.into()), Arc::new(()));
        assert_eq!(image.frame_count(), 1);
        assert_eq!(image.size(0), size(8u32.into(), 6u32.into()));
        assert_eq!(image.as_bytes(0), None);
        assert!(image.external().is_some());
    }
}
