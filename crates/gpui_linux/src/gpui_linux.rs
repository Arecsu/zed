#![cfg(any(target_os = "linux", target_os = "freebsd"))]
mod linux;

use gpui::{DevicePixels, GpuSpecs, PlatformAtlas, Scene, Size};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use std::sync::{Arc, OnceLock};

/// The renderer boundary used by Zoe's private Vulkan presentation adapter.
/// GPUI owns scene scheduling and the platform window; the application-owned
/// implementation owns the Vulkan device, surface resources, and submission
/// ordering. This first contract intentionally exposes only the operations
/// needed by the surface smoke slice.
pub trait VulkanRenderer: 'static {
    fn max_texture_size(&self) -> u32;
    fn gpu_specs(&self) -> GpuSpecs;
    fn draw(&mut self, scene: &Scene) -> bool;
    fn needs_redraw(&mut self) -> bool;
    fn device_lost(&self) -> bool;
    fn recover(&mut self) -> anyhow::Result<()>;
    fn update_drawable_size(&mut self, size: Size<DevicePixels>);
    fn update_transparency(&mut self, transparent: bool);
    fn set_subpixel_layout(&mut self, is_bgr: bool);
    fn sprite_atlas(&self) -> Arc<dyn PlatformAtlas>;
    fn destroy(&mut self);
}

/// Factory installed before `gpui_platform::application()` creates the
/// Wayland client. The raw handles belong to the GPUI window and remain valid
/// for the returned renderer's lifetime.
pub trait VulkanRendererFactory: Send + Sync + 'static {
    fn create(
        &self,
        display: RawDisplayHandle,
        window: RawWindowHandle,
        size: Size<DevicePixels>,
        transparent: bool,
    ) -> anyhow::Result<Box<dyn VulkanRenderer>>;
}

static VULKAN_RENDERER_FACTORY: OnceLock<Arc<dyn VulkanRendererFactory>> = OnceLock::new();

pub fn set_vulkan_renderer_factory(
    factory: Arc<dyn VulkanRendererFactory>,
) -> Result<(), Arc<dyn VulkanRendererFactory>> {
    VULKAN_RENDERER_FACTORY.set(factory)
}

pub(crate) fn vulkan_renderer_factory() -> Option<Arc<dyn VulkanRendererFactory>> {
    VULKAN_RENDERER_FACTORY.get().cloned()
}

pub use linux::current_platform;
