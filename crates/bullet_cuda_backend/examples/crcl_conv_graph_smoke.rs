//! CRCL-v2 Conv2d を acyclib graph IR 経由で実行する smoke
//!
//! 直接 NVRTC を呼んでいた crcl_conv_smoke を、 GraphBuilder + Conv2d op 経由で実行する形に
//! 書き直したもの。 graph IR から CudaDevice 上で Conv2d kernel が動くことを確認する。
//!
//! 実行:
//!   cargo run --release --example crcl_conv_graph_smoke
use acyclib::{
    device::{Device, tensor::Shape},
    graph::{
        GraphNodeId, GraphNodeIdTy,
        builder::{GraphBuilder, InitSettings},
    },
};
use bullet_cuda_backend::{CudaDevice, ops::Conv2d};

fn main() {
    println!("=== CRCL-v2 Conv2d graph IR smoke ===");

    let cin = 192i32;
    let cout = 192i32;
    let h = 9i32;
    let w = 9i32;
    let k = 3i32;
    let pad = 1i32;
    let stride = 1i32;
    let hout = (h + 2 * pad - k) / stride + 1;
    let wout = (w + 2 * pad - k) / stride + 1;

    let builder = GraphBuilder::default();
    let input = builder.new_dense_input("input", Shape::new((cin * h * w) as usize, 1));
    let weight = builder.new_weights("weight", Shape::new((cout * cin * k * k) as usize, 1), InitSettings::Zeroed);

    let conv = builder.apply(Conv2d {
        input: input.annotated_node(),
        weight: weight.annotated_node(),
        in_channels: cin,
        out_channels: cout,
        spatial_h: h,
        spatial_w: w,
        kernel_size: k,
        padding: pad,
        stride,
    });
    let out_idx = conv.annotated_node().idx;

    // graph compile は root が (1, 1) not batched を要求するので、
    // ones (1, Cout*Hout*Wout) と matmul して scalar に reduce。
    // builder.build() 内の optimise が自動で ReduceAcrossBatch を追加してくれる。
    let out_size = (cout * hout * wout) as usize;
    let ones = builder.new_constant(Shape::new(1, out_size), &vec![1.0f32; out_size]);
    let _scalar = ones.matmul(conv);

    let device = CudaDevice::new(0).expect("CudaDevice::new");
    let mut graph = builder.build(device);
    println!("graph built");

    // weight に all-ones を書き込む
    {
        let w_ref = graph.get_weights("weight");
        let mut w_dense = w_ref.dense_mut();
        let nw = (cout * cin * k * k) as usize;
        w_dense.load_from_slice(None, &vec![1.0f32; nw]).expect("weight load");
    }

    // input に batch_size=1 で all-ones を書き込む
    {
        let in_ref = graph.get_input("input");
        let mut in_dense = in_ref.dense_mut();
        let nx = (cin * h * w) as usize;
        in_dense.load_from_slice(Some(1), &vec![1.0f32; nx]).expect("input load");
    }

    // forward 実行 (root が scalar でないので graph.forward() は使えない)
    graph.execute("forward").expect("forward exec");
    graph.synchronise().expect("synchronise");

    // out_idx の値を読み出して center=1728 を確認
    let out_ref = graph.get(GraphNodeId::new(out_idx, GraphNodeIdTy::Values)).expect("get out");
    let out_dense = out_ref.dense();
    let n_out = (cout * hout * wout) as usize;
    let mut host = vec![0.0f32; n_out];
    out_dense.write_to_slice(&mut host).expect("write_to_slice");

    let center = host[((cout / 2 * hout + hout / 2) * wout + wout / 2) as usize];
    let corner = host[0];
    let edge = host[(wout / 2) as usize];
    println!("forward via graph IR: center={center} corner={corner} edge={edge}");
    let expected_center = (cin * k * k) as f32;
    let expected_corner = (cin * 4) as f32;
    let expected_edge = (cin * 6) as f32;
    assert!((center - expected_center).abs() < 1e-3);
    assert!((corner - expected_corner).abs() < 1e-3);
    assert!((edge - expected_edge).abs() < 1e-3);
    println!("Conv2d graph IR smoke PASSED");
}
