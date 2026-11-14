from railway.gen.seeds import candidate_seeds, fresh_seeds, ranges


def test_T04_25_seed_sets_are_ordered_and_disjoint():
    dev = list(candidate_seeds("dev", count=100))
    train = list(candidate_seeds("train", count=100))
    heldout = list(candidate_seeds("heldout", count=100))
    assert dev == sorted(dev) and train == sorted(train) and heldout == sorted(heldout)
    assert not set(dev) & set(train)
    assert not set(dev) & set(heldout)
    assert not set(train) & set(heldout)
    assert all(seed < 10_000 for seed in dev)
    assert all(10_000 <= seed < 1_000_000 for seed in train)
    assert all(1_000_000 <= seed < 1_010_000 for seed in heldout)
    assert min(fresh_seeds(100, seed=5)) >= 2_000_000
    assert ranges() == {
        "dev": (0, 10_000),
        "train": (10_000, 1_000_000),
        "heldout": (1_000_000, 1_010_000),
        "fresh": (2_000_000, 2**31),
    }
