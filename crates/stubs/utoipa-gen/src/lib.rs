use proc_macro::TokenStream;

#[proc_macro_derive(OpenApi, attributes(openapi))]
pub fn derive_openapi(input: TokenStream) -> TokenStream {
    let ast: syn::DeriveInput = match syn::parse(input) {
        Ok(ast) => ast,
        Err(_) => return TokenStream::new(),
    };
    let name = ast.ident;
    let expanded = quote::quote! {
        impl utoipa::OpenApi for #name {
            fn openapi() -> utoipa::openapi::OpenApi {
                utoipa::openapi::OpenApi::new("GFlowNet-Prover API", env!("CARGO_PKG_VERSION"))
            }
        }
    };
    TokenStream::from(expanded)
}

#[proc_macro_derive(ToSchema, attributes(schema))]
pub fn derive_to_schema(input: TokenStream) -> TokenStream {
    let ast: syn::DeriveInput = match syn::parse(input) {
        Ok(ast) => ast,
        Err(_) => return TokenStream::new(),
    };
    let name = ast.ident;
    let name_str = name.to_string();
    let expanded = quote::quote! {
        impl utoipa::ToSchema for #name {
            fn name() -> std::borrow::Cow<'static, str> {
                std::borrow::Cow::Borrowed(#name_str)
            }
        }
    };
    TokenStream::from(expanded)
}

#[proc_macro_attribute]
pub fn path(_attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}
