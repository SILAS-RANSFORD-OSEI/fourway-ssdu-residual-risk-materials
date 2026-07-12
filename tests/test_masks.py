from fourway_mri.masks import generate_fourway_partition


def test_fourway_partition_validity_width_320():
    partition = generate_fourway_partition(
        width=320,
        volume_id="test_volume_320",
        base_seed=123,
        target_acceleration=4.0,
        acs_lines=24,
        density_power=4.0,
        theta_outer_fraction=0.50,
        lambda_rec_fraction=0.25,
        lambda_risk_fraction=0.15,
        lambda_hold_fraction=0.10,
    )

    assert partition["pairwise_overlap_count"] == 0
    assert partition["union_matches_omega"] is True
    assert partition["acs_subset_of_theta"] is True
    assert partition["valid_index_bounds"] is True
    assert partition["count_omega"] == round(320 / 4.0)


def test_fourway_partition_validity_width_396():
    partition = generate_fourway_partition(
        width=396,
        volume_id="test_volume_396",
        base_seed=123,
        target_acceleration=4.0,
        acs_lines=24,
        density_power=4.0,
        theta_outer_fraction=0.50,
        lambda_rec_fraction=0.25,
        lambda_risk_fraction=0.15,
        lambda_hold_fraction=0.10,
    )

    assert partition["pairwise_overlap_count"] == 0
    assert partition["union_matches_omega"] is True
    assert partition["acs_subset_of_theta"] is True
    assert partition["valid_index_bounds"] is True
    assert partition["count_omega"] == round(396 / 4.0)


def test_fourway_partition_validity_width_272():
    partition = generate_fourway_partition(
        width=272,
        volume_id="test_volume_272",
        base_seed=123,
        target_acceleration=4.0,
        acs_lines=24,
        density_power=4.0,
        theta_outer_fraction=0.50,
        lambda_rec_fraction=0.25,
        lambda_risk_fraction=0.15,
        lambda_hold_fraction=0.10,
    )

    assert partition["pairwise_overlap_count"] == 0
    assert partition["union_matches_omega"] is True
    assert partition["acs_subset_of_theta"] is True
    assert partition["valid_index_bounds"] is True
    assert partition["count_omega"] == round(272 / 4.0)
