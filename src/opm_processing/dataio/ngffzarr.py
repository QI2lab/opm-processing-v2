import acquire_zarr as aqz

def create_via_acquire(output_path,dimension_shape):
    settings = aqz.StreamSettings(
        store_path=str(output_path),
        data_type=aqz.DataType.UINT16,
        version=aqz.ZarrVersion.V3,
        multiscale=False
    )
    
    settings.compression = aqz.CompressionSettings(
        compressor=aqz.Compressor.BLOSC1,
        codec=aqz.CompressionCodec.BLOSC_LZ4,
        level=4,
        shuffle=2,
    )

    settings.dimensions.extend([
        aqz.Dimension(
            name="t",
            type=aqz.DimensionType.TIME,
            array_size_px=dimension_shape[0],
            chunk_size_px=1,
            shard_size_chunks=1
        ),
        aqz.Dimension(
            name="p",
            type=aqz.DimensionType.SPACE,
            array_size_px=dimension_shape[1],
            chunk_size_px=1,
            shard_size_chunks=1
        ),
        aqz.Dimension(
            name="c",
            type=aqz.DimensionType.CHANNEL,
            array_size_px=dimension_shape[2],
            chunk_size_px=1,
            shard_size_chunks=1
        ),
        aqz.Dimension(
            name="z",
            type=aqz.DimensionType.SPACE,
            array_size_px=dimension_shape[3],
            chunk_size_px=256,
            shard_size_chunks=1
        ),
        aqz.Dimension(
            name="y",
            type=aqz.DimensionType.SPACE,
            array_size_px=dimension_shape[4],
            chunk_size_px=dimension_shape[4]//4,
            shard_size_chunks=2
        ),
        aqz.Dimension(
            name="x",
            type=aqz.DimensionType.SPACE,
            array_size_px=dimension_shape[5],
            chunk_size_px=dimension_shape[5]//4,
            shard_size_chunks=2
        )
    ])
    stream = aqz.ZarrStream(settings)

    return stream

def write_via_acquire(stream,data):
    
    stream.append(data)