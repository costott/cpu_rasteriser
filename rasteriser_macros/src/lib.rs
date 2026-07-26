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
            _ => panic!("Interpolate only supports named structs"),
        },
        _ => panic!("Interpolate only supports structs"),
    };

    let interpolate_fields = fields.iter().map(|field| {
        let name = field.ident.as_ref().unwrap();

        quote! {
            #name: self.#name.interpolate(&other.#name, t)
        }
    });

    let expanded = quote! {
        impl Interpolate for #name {
            fn interpolate(
                &self,
                other: &Self,
                t: f32,
            ) -> Self {
                Self {
                    #(#interpolate_fields),*
                }
            }
        }
    };

    TokenStream::from(expanded)
}
