use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, parse_macro_input};

#[proc_macro_derive(Interpolate)]
pub fn derive_interpolate(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    let name = input.ident;

    let fields = match input.data {
        Data::Struct(data) => match data.fields {
            Fields::Named(fields) => fields.named,
            Fields::Unnamed(_) => {
                return syn::Error::new_spanned(name, "Interpolate only supports named fields")
                    .to_compile_error()
                    .into();
            }
            Fields::Unit => {
                return syn::Error::new_spanned(
                    name,
                    "Interpolate cannot be derived for unit structs",
                )
                .to_compile_error()
                .into();
            }
        },

        _ => {
            return syn::Error::new_spanned(name, "Interpolate can only be derived for structs")
                .to_compile_error()
                .into();
        }
    };

    let field_names: Vec<_> = fields
        .iter()
        .map(|field| field.ident.as_ref().unwrap())
        .collect();

    let expanded = quote! {
        impl Interpolate for #name {
            fn interpolate(&self, other: &Self, t: f32) -> Self {
                Self {
                    #(
                        #field_names: self.#field_names.interpolate(&other.#field_names, t),
                    )*
                }
            }

            fn difference(&self, other: &Self) -> Self {
                Self {
                    #(
                        #field_names: self.#field_names.difference(&other.#field_names),
                    )*
                }
            }

            fn scale(&self, factor: f32) -> Self {
                Self {
                    #(
                        #field_names: self.#field_names.scale(factor),
                    )*
                }
            }

            fn add_scaled(&self, other: &Self, factor: f32) -> Self {
                Self {
                    #(
                        #field_names: self.#field_names.add_scaled(&other.#field_names, factor),
                    )*
                }
            }
        }

        impl Clone for #name {
            fn clone(&self) -> Self {
                Self {
                    #(
                        #field_names: self.#field_names.clone(),
                    )*
                }
            }
        }
    };

    expanded.into()
}
