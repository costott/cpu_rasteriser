use crate::graphics::fragment::Fragment;

pub trait FragmentShader {
    fn shade(&self, fragment: Fragment) -> Option<Fragment>;
}

pub struct BasicFragmentShader;
impl FragmentShader for BasicFragmentShader {
    fn shade(&self, fragment: Fragment) -> Option<Fragment> {
        Some(fragment)
    }
}
