def test_package_import():
    import fourway_mri

    assert hasattr(fourway_mri, "__version__")
