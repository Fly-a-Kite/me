from datadiff.seed_budget import SeedBudgetAllocator


def test_seed_budget_allocator_applies_tier_multiplier():
    allocator = SeedBudgetAllocator(tier_selector=lambda index: "high")

    energy = allocator.energy_for_seed(
        0,
        pulls=0,
        mean_reward=2.0,
        family_breadth=3,
        cluster_outcome_count=2,
        since_last_finding_pulls=0,
    )

    assert energy >= 2


def test_seed_budget_allocator_bounds_energy():
    allocator = SeedBudgetAllocator(tier_selector=lambda index: "med", min_energy=1, max_energy=4)

    assert allocator.bounded_energy(0) == 1
    assert allocator.bounded_energy(3) == 3
    assert allocator.bounded_energy(99) == 4
