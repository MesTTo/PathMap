use crate::PathMap;
use crate::trie_node::{TaggedNodeRef, TrieNode};
use crate::zipper::*;
#[cfg(not(test))]
use core::sync::atomic::{AtomicUsize, Ordering::Relaxed};

#[cfg(not(test))]
static MAKE_UNIQUE_CALLS: AtomicUsize = AtomicUsize::new(0);
#[cfg(not(test))]
static COW_CLONES: AtomicUsize = AtomicUsize::new(0);
#[cfg(not(test))]
static MATERIALIZE_CALLS: AtomicUsize = AtomicUsize::new(0);
#[cfg(not(test))]
static MATERIALIZED_SCOUT_FRAMES: AtomicUsize = AtomicUsize::new(0);
#[cfg(not(test))]
static PRODUCT_FACTOR_ENTRIES: AtomicUsize = AtomicUsize::new(0);
#[cfg(not(test))]
static DEPENDENT_PRODUCT_ENROLL_CALLS: AtomicUsize = AtomicUsize::new(0);
#[cfg(not(test))]
static DEPENDENT_PRODUCT_FACTOR_ENTRIES: AtomicUsize = AtomicUsize::new(0);
#[cfg(not(test))]
static OVERLAY_VALUE_MAPS: AtomicUsize = AtomicUsize::new(0);
#[cfg(not(test))]
static OVERLAY_CHILD_MASK_UNIONS: AtomicUsize = AtomicUsize::new(0);
#[cfg(not(test))]
static RESTRICT_VALUE_FILTERS: AtomicUsize = AtomicUsize::new(0);
#[cfg(not(test))]
static RESTRICT_CHILD_MASK_FILTERS: AtomicUsize = AtomicUsize::new(0);
#[cfg(not(test))]
static SUBTRACT_VALUE_MAPS: AtomicUsize = AtomicUsize::new(0);
#[cfg(not(test))]
static SUBTRACT_CHILD_PROBES: AtomicUsize = AtomicUsize::new(0);
#[cfg(not(test))]
static DIFF_OBSERVATION_CHECKS: AtomicUsize = AtomicUsize::new(0);
#[cfg(not(test))]
static OBSERVER_MATERIALIZATIONS: AtomicUsize = AtomicUsize::new(0);
#[cfg(not(test))]
static OBSERVER_MATERIALIZED_PATHS: AtomicUsize = AtomicUsize::new(0);
#[cfg(not(test))]
static OBSERVER_MATERIALIZED_VALUES: AtomicUsize = AtomicUsize::new(0);

#[cfg(test)]
std::thread_local! {
    static TEST_MAKE_UNIQUE_CALLS: std::cell::Cell<usize> = std::cell::Cell::new(0);
    static TEST_COW_CLONES: std::cell::Cell<usize> = std::cell::Cell::new(0);
    static TEST_MATERIALIZE_CALLS: std::cell::Cell<usize> = std::cell::Cell::new(0);
    static TEST_MATERIALIZED_SCOUT_FRAMES: std::cell::Cell<usize> = std::cell::Cell::new(0);
    static TEST_PRODUCT_FACTOR_ENTRIES: std::cell::Cell<usize> = std::cell::Cell::new(0);
    static TEST_DEPENDENT_PRODUCT_ENROLL_CALLS: std::cell::Cell<usize> = std::cell::Cell::new(0);
    static TEST_DEPENDENT_PRODUCT_FACTOR_ENTRIES: std::cell::Cell<usize> = std::cell::Cell::new(0);
    static TEST_OVERLAY_VALUE_MAPS: std::cell::Cell<usize> = std::cell::Cell::new(0);
    static TEST_OVERLAY_CHILD_MASK_UNIONS: std::cell::Cell<usize> = std::cell::Cell::new(0);
    static TEST_RESTRICT_VALUE_FILTERS: std::cell::Cell<usize> = std::cell::Cell::new(0);
    static TEST_RESTRICT_CHILD_MASK_FILTERS: std::cell::Cell<usize> = std::cell::Cell::new(0);
    static TEST_SUBTRACT_VALUE_MAPS: std::cell::Cell<usize> = std::cell::Cell::new(0);
    static TEST_SUBTRACT_CHILD_PROBES: std::cell::Cell<usize> = std::cell::Cell::new(0);
    static TEST_DIFF_OBSERVATION_CHECKS: std::cell::Cell<usize> = std::cell::Cell::new(0);
    static TEST_OBSERVER_MATERIALIZATIONS: std::cell::Cell<usize> = std::cell::Cell::new(0);
    static TEST_OBSERVER_MATERIALIZED_PATHS: std::cell::Cell<usize> = std::cell::Cell::new(0);
    static TEST_OBSERVER_MATERIALIZED_VALUES: std::cell::Cell<usize> = std::cell::Cell::new(0);
}

#[cfg(test)]
static COUNTER_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
pub(crate) fn counter_test_guard() -> std::sync::MutexGuard<'static, ()> {
    COUNTER_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CowCloneCounters {
    pub make_unique_calls: usize,
    pub cow_clones: usize,
    pub materialize_calls: usize,
    pub materialized_scout_frames: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct VirtualZipperCounters {
    pub product_factor_entries: usize,
    pub dependent_product_enroll_calls: usize,
    pub dependent_product_factor_entries: usize,
    pub overlay_value_maps: usize,
    pub overlay_child_mask_unions: usize,
    pub restrict_value_filters: usize,
    pub restrict_child_mask_filters: usize,
    pub subtract_value_maps: usize,
    pub subtract_child_probes: usize,
    pub diff_observation_checks: usize,
    pub observer_materializations: usize,
    pub observer_materialized_paths: usize,
    pub observer_materialized_values: usize,
}

pub(crate) fn record_make_unique(cloned: bool) {
    #[cfg(test)]
    {
        TEST_MAKE_UNIQUE_CALLS.with(|counter| counter.set(counter.get() + 1));
        if cloned {
            TEST_COW_CLONES.with(|counter| counter.set(counter.get() + 1));
        }
        return;
    }

    #[cfg(not(test))]
    {
        MAKE_UNIQUE_CALLS.fetch_add(1, Relaxed);
        if cloned {
            COW_CLONES.fetch_add(1, Relaxed);
        }
    }
}

pub(crate) fn record_materialize_for_write(scout_depth: usize) {
    #[cfg(test)]
    {
        TEST_MATERIALIZE_CALLS.with(|counter| counter.set(counter.get() + 1));
        TEST_MATERIALIZED_SCOUT_FRAMES.with(|counter| counter.set(counter.get() + scout_depth));
        return;
    }

    #[cfg(not(test))]
    {
        MATERIALIZE_CALLS.fetch_add(1, Relaxed);
        MATERIALIZED_SCOUT_FRAMES.fetch_add(scout_depth, Relaxed);
    }
}

pub(crate) fn record_product_factor_entry() {
    #[cfg(test)]
    {
        TEST_PRODUCT_FACTOR_ENTRIES.with(|counter| counter.set(counter.get() + 1));
        return;
    }

    #[cfg(not(test))]
    {
        PRODUCT_FACTOR_ENTRIES.fetch_add(1, Relaxed);
    }
}

pub(crate) fn record_dependent_product_enroll_call() {
    #[cfg(test)]
    {
        TEST_DEPENDENT_PRODUCT_ENROLL_CALLS.with(|counter| counter.set(counter.get() + 1));
        return;
    }

    #[cfg(not(test))]
    {
        DEPENDENT_PRODUCT_ENROLL_CALLS.fetch_add(1, Relaxed);
    }
}

pub(crate) fn record_dependent_product_factor_entry() {
    #[cfg(test)]
    {
        TEST_DEPENDENT_PRODUCT_FACTOR_ENTRIES.with(|counter| counter.set(counter.get() + 1));
        return;
    }

    #[cfg(not(test))]
    {
        DEPENDENT_PRODUCT_FACTOR_ENTRIES.fetch_add(1, Relaxed);
    }
}

pub(crate) fn record_overlay_value_map() {
    #[cfg(test)]
    {
        TEST_OVERLAY_VALUE_MAPS.with(|counter| counter.set(counter.get() + 1));
        return;
    }

    #[cfg(not(test))]
    {
        OVERLAY_VALUE_MAPS.fetch_add(1, Relaxed);
    }
}

pub(crate) fn record_overlay_child_mask_union() {
    #[cfg(test)]
    {
        TEST_OVERLAY_CHILD_MASK_UNIONS.with(|counter| counter.set(counter.get() + 1));
        return;
    }

    #[cfg(not(test))]
    {
        OVERLAY_CHILD_MASK_UNIONS.fetch_add(1, Relaxed);
    }
}

pub(crate) fn record_restrict_value_filter() {
    #[cfg(test)]
    {
        TEST_RESTRICT_VALUE_FILTERS.with(|counter| counter.set(counter.get() + 1));
        return;
    }

    #[cfg(not(test))]
    {
        RESTRICT_VALUE_FILTERS.fetch_add(1, Relaxed);
    }
}

pub(crate) fn record_restrict_child_mask_filter() {
    #[cfg(test)]
    {
        TEST_RESTRICT_CHILD_MASK_FILTERS.with(|counter| counter.set(counter.get() + 1));
        return;
    }

    #[cfg(not(test))]
    {
        RESTRICT_CHILD_MASK_FILTERS.fetch_add(1, Relaxed);
    }
}

pub(crate) fn record_subtract_value_map() {
    #[cfg(test)]
    {
        TEST_SUBTRACT_VALUE_MAPS.with(|counter| counter.set(counter.get() + 1));
        return;
    }

    #[cfg(not(test))]
    {
        SUBTRACT_VALUE_MAPS.fetch_add(1, Relaxed);
    }
}

pub(crate) fn record_subtract_child_probe() {
    #[cfg(test)]
    {
        TEST_SUBTRACT_CHILD_PROBES.with(|counter| counter.set(counter.get() + 1));
        return;
    }

    #[cfg(not(test))]
    {
        SUBTRACT_CHILD_PROBES.fetch_add(1, Relaxed);
    }
}

pub(crate) fn record_diff_observation_check() {
    #[cfg(test)]
    {
        TEST_DIFF_OBSERVATION_CHECKS.with(|counter| counter.set(counter.get() + 1));
        return;
    }

    #[cfg(not(test))]
    {
        DIFF_OBSERVATION_CHECKS.fetch_add(1, Relaxed);
    }
}

pub(crate) fn record_observer_materialization() {
    #[cfg(test)]
    {
        TEST_OBSERVER_MATERIALIZATIONS.with(|counter| counter.set(counter.get() + 1));
        return;
    }

    #[cfg(not(test))]
    {
        OBSERVER_MATERIALIZATIONS.fetch_add(1, Relaxed);
    }
}

pub(crate) fn record_observer_materialized_path(has_value: bool) {
    #[cfg(test)]
    {
        TEST_OBSERVER_MATERIALIZED_PATHS.with(|counter| counter.set(counter.get() + 1));
        if has_value {
            TEST_OBSERVER_MATERIALIZED_VALUES.with(|counter| counter.set(counter.get() + 1));
        }
        return;
    }

    #[cfg(not(test))]
    {
        OBSERVER_MATERIALIZED_PATHS.fetch_add(1, Relaxed);
        if has_value {
            OBSERVER_MATERIALIZED_VALUES.fetch_add(1, Relaxed);
        }
    }
}

pub fn reset_cow_clone_counters() {
    #[cfg(test)]
    {
        TEST_MAKE_UNIQUE_CALLS.with(|counter| counter.set(0));
        TEST_COW_CLONES.with(|counter| counter.set(0));
        TEST_MATERIALIZE_CALLS.with(|counter| counter.set(0));
        TEST_MATERIALIZED_SCOUT_FRAMES.with(|counter| counter.set(0));
        return;
    }

    #[cfg(not(test))]
    {
        MAKE_UNIQUE_CALLS.store(0, Relaxed);
        COW_CLONES.store(0, Relaxed);
        MATERIALIZE_CALLS.store(0, Relaxed);
        MATERIALIZED_SCOUT_FRAMES.store(0, Relaxed);
    }
}

pub fn cow_clone_counters() -> CowCloneCounters {
    #[cfg(test)]
    {
        return CowCloneCounters {
            make_unique_calls: TEST_MAKE_UNIQUE_CALLS.with(|counter| counter.get()),
            cow_clones: TEST_COW_CLONES.with(|counter| counter.get()),
            materialize_calls: TEST_MATERIALIZE_CALLS.with(|counter| counter.get()),
            materialized_scout_frames: TEST_MATERIALIZED_SCOUT_FRAMES.with(|counter| counter.get()),
        };
    }

    #[cfg(not(test))]
    {
        CowCloneCounters {
            make_unique_calls: MAKE_UNIQUE_CALLS.load(Relaxed),
            cow_clones: COW_CLONES.load(Relaxed),
            materialize_calls: MATERIALIZE_CALLS.load(Relaxed),
            materialized_scout_frames: MATERIALIZED_SCOUT_FRAMES.load(Relaxed),
        }
    }
}

pub fn reset_virtual_zipper_counters() {
    #[cfg(test)]
    {
        TEST_PRODUCT_FACTOR_ENTRIES.with(|counter| counter.set(0));
        TEST_DEPENDENT_PRODUCT_ENROLL_CALLS.with(|counter| counter.set(0));
        TEST_DEPENDENT_PRODUCT_FACTOR_ENTRIES.with(|counter| counter.set(0));
        TEST_OVERLAY_VALUE_MAPS.with(|counter| counter.set(0));
        TEST_OVERLAY_CHILD_MASK_UNIONS.with(|counter| counter.set(0));
        TEST_RESTRICT_VALUE_FILTERS.with(|counter| counter.set(0));
        TEST_RESTRICT_CHILD_MASK_FILTERS.with(|counter| counter.set(0));
        TEST_SUBTRACT_VALUE_MAPS.with(|counter| counter.set(0));
        TEST_SUBTRACT_CHILD_PROBES.with(|counter| counter.set(0));
        TEST_DIFF_OBSERVATION_CHECKS.with(|counter| counter.set(0));
        TEST_OBSERVER_MATERIALIZATIONS.with(|counter| counter.set(0));
        TEST_OBSERVER_MATERIALIZED_PATHS.with(|counter| counter.set(0));
        TEST_OBSERVER_MATERIALIZED_VALUES.with(|counter| counter.set(0));
        return;
    }

    #[cfg(not(test))]
    {
        PRODUCT_FACTOR_ENTRIES.store(0, Relaxed);
        DEPENDENT_PRODUCT_ENROLL_CALLS.store(0, Relaxed);
        DEPENDENT_PRODUCT_FACTOR_ENTRIES.store(0, Relaxed);
        OVERLAY_VALUE_MAPS.store(0, Relaxed);
        OVERLAY_CHILD_MASK_UNIONS.store(0, Relaxed);
        RESTRICT_VALUE_FILTERS.store(0, Relaxed);
        RESTRICT_CHILD_MASK_FILTERS.store(0, Relaxed);
        SUBTRACT_VALUE_MAPS.store(0, Relaxed);
        SUBTRACT_CHILD_PROBES.store(0, Relaxed);
        DIFF_OBSERVATION_CHECKS.store(0, Relaxed);
        OBSERVER_MATERIALIZATIONS.store(0, Relaxed);
        OBSERVER_MATERIALIZED_PATHS.store(0, Relaxed);
        OBSERVER_MATERIALIZED_VALUES.store(0, Relaxed);
    }
}

pub fn virtual_zipper_counters() -> VirtualZipperCounters {
    #[cfg(test)]
    {
        return VirtualZipperCounters {
            product_factor_entries: TEST_PRODUCT_FACTOR_ENTRIES.with(|counter| counter.get()),
            dependent_product_enroll_calls: TEST_DEPENDENT_PRODUCT_ENROLL_CALLS
                .with(|counter| counter.get()),
            dependent_product_factor_entries: TEST_DEPENDENT_PRODUCT_FACTOR_ENTRIES
                .with(|counter| counter.get()),
            overlay_value_maps: TEST_OVERLAY_VALUE_MAPS.with(|counter| counter.get()),
            overlay_child_mask_unions: TEST_OVERLAY_CHILD_MASK_UNIONS.with(|counter| counter.get()),
            restrict_value_filters: TEST_RESTRICT_VALUE_FILTERS.with(|counter| counter.get()),
            restrict_child_mask_filters: TEST_RESTRICT_CHILD_MASK_FILTERS
                .with(|counter| counter.get()),
            subtract_value_maps: TEST_SUBTRACT_VALUE_MAPS.with(|counter| counter.get()),
            subtract_child_probes: TEST_SUBTRACT_CHILD_PROBES.with(|counter| counter.get()),
            diff_observation_checks: TEST_DIFF_OBSERVATION_CHECKS.with(|counter| counter.get()),
            observer_materializations: TEST_OBSERVER_MATERIALIZATIONS.with(|counter| counter.get()),
            observer_materialized_paths: TEST_OBSERVER_MATERIALIZED_PATHS
                .with(|counter| counter.get()),
            observer_materialized_values: TEST_OBSERVER_MATERIALIZED_VALUES
                .with(|counter| counter.get()),
        };
    }

    #[cfg(not(test))]
    {
        VirtualZipperCounters {
            product_factor_entries: PRODUCT_FACTOR_ENTRIES.load(Relaxed),
            dependent_product_enroll_calls: DEPENDENT_PRODUCT_ENROLL_CALLS.load(Relaxed),
            dependent_product_factor_entries: DEPENDENT_PRODUCT_FACTOR_ENTRIES.load(Relaxed),
            overlay_value_maps: OVERLAY_VALUE_MAPS.load(Relaxed),
            overlay_child_mask_unions: OVERLAY_CHILD_MASK_UNIONS.load(Relaxed),
            restrict_value_filters: RESTRICT_VALUE_FILTERS.load(Relaxed),
            restrict_child_mask_filters: RESTRICT_CHILD_MASK_FILTERS.load(Relaxed),
            subtract_value_maps: SUBTRACT_VALUE_MAPS.load(Relaxed),
            subtract_child_probes: SUBTRACT_CHILD_PROBES.load(Relaxed),
            diff_observation_checks: DIFF_OBSERVATION_CHECKS.load(Relaxed),
            observer_materializations: OBSERVER_MATERIALIZATIONS.load(Relaxed),
            observer_materialized_paths: OBSERVER_MATERIALIZED_PATHS.load(Relaxed),
            observer_materialized_values: OBSERVER_MATERIALIZED_VALUES.load(Relaxed),
        }
    }
}

/// Example usage of counters
///
/// ```
/// pathmap::counters::print_traversal(&map.read_zipper());
/// let counters = pathmap::counters::Counters::count_ocupancy(&map);
/// counters.print_histogram_by_depth();
/// counters.print_run_length_histogram();
/// counters.print_list_node_stats();
/// ```
pub struct Counters {
    total_nodes_by_depth: Vec<usize>,
    total_child_items_by_depth: Vec<usize>,
    max_child_items_by_depth: Vec<usize>,

    /// Counts the number of each node type at a given depth
    total_dense_byte_nodes_by_depth: Vec<usize>,
    total_list_nodes_by_depth: Vec<usize>,

    /// List-node-specific counters
    total_slot0_length_by_depth: Vec<usize>,
    slot1_occupancy_count_by_depth: Vec<usize>,
    total_slot1_length_by_depth: Vec<usize>,
    list_node_single_byte_keys_by_depth: Vec<usize>,

    /// Counts the runs of distance (in bytes) that end at each byte depth
    /// [run_length][ending_byte_depth]
    run_length_histogram_by_ending_byte_depth: Vec<Vec<usize>>,
    cur_run_start_depth: usize,
}
impl Counters {
    pub const fn new() -> Self {
        Self {
            total_nodes_by_depth: vec![],
            total_child_items_by_depth: vec![],
            max_child_items_by_depth: vec![],
            total_dense_byte_nodes_by_depth: vec![],
            total_list_nodes_by_depth: vec![],
            total_slot0_length_by_depth: vec![],
            slot1_occupancy_count_by_depth: vec![],
            total_slot1_length_by_depth: vec![],
            list_node_single_byte_keys_by_depth: vec![],
            run_length_histogram_by_ending_byte_depth: vec![],
            cur_run_start_depth: 0,
        }
    }
    pub fn total_nodes(&self) -> usize {
        let mut total = 0;
        self.total_nodes_by_depth
            .iter()
            .for_each(|cnt| total += cnt);
        total
    }
    pub fn total_child_items(&self) -> usize {
        let mut total = 0;
        self.total_child_items_by_depth
            .iter()
            .for_each(|cnt| total += cnt);
        total
    }
    pub fn print_histogram_by_depth(&self) {
        println!(
            "\n\ttotal_nodes\ttot_child_cnt\tavg_branch\tmax_child_items\tdense_nodes\tlist_nodes"
        );
        for depth in 0..self.total_nodes_by_depth.len() {
            println!(
                "{depth}\t{}\t\t{}\t\t{:1.4}\t\t{}\t\t{}\t\t{}",
                self.total_nodes_by_depth[depth],
                self.total_child_items_by_depth[depth],
                self.total_child_items_by_depth[depth] as f32
                    / self.total_nodes_by_depth[depth] as f32,
                self.max_child_items_by_depth[depth],
                self.total_dense_byte_nodes_by_depth[depth],
                self.total_list_nodes_by_depth[depth],
            );
        }
        println!(
            "TOTAL nodes: {}, items: {}, avg children-per-node: {}",
            self.total_nodes(),
            self.total_child_items(),
            self.total_child_items() as f32 / self.total_nodes() as f32
        );
    }
    pub fn print_run_length_histogram(&self) {
        println!("run_len\trun_cnt\trun_end_mean_depth");
        for (run_length, depths) in self
            .run_length_histogram_by_ending_byte_depth
            .iter()
            .enumerate()
        {
            let total = depths.iter().fold(0, |mut sum, cnt| {
                sum += cnt;
                sum
            });
            let depth_sum = depths.iter().enumerate().fold(0, |mut sum, (depth, cnt)| {
                sum += cnt * (depth + 1);
                sum
            });
            println!("{run_length}\t{total}\t{}", depth_sum as f32 / total as f32);
        }
    }
    pub fn print_list_node_stats(&self) {
        println!(
            "\n\ttotal_nodes\tlist_node_cnt\tlist_node_rto\tavg_slot0_len\tslot1_cnt\tslot1_used_rto\tavg_slot1_len\tone_byte_keys\tone_byte_rto"
        );
        for depth in 0..self.total_nodes_by_depth.len() {
            println!(
                "{depth}\t{}\t\t{}\t\t{:2.1}%\t\t{:1.4}\t\t{}\t\t{:2.1}%\t\t{:1.4}\t\t{}\t\t{:2.1}%",
                self.total_nodes_by_depth[depth],
                self.total_list_nodes_by_depth[depth],
                self.total_list_nodes_by_depth[depth] as f32
                    / self.total_nodes_by_depth[depth] as f32
                    * 100.0,
                self.total_slot0_length_by_depth[depth] as f32
                    / self.total_list_nodes_by_depth[depth] as f32,
                self.slot1_occupancy_count_by_depth[depth],
                self.slot1_occupancy_count_by_depth[depth] as f32
                    / self.total_list_nodes_by_depth[depth] as f32
                    * 100.0,
                self.total_slot1_length_by_depth[depth] as f32
                    / self.slot1_occupancy_count_by_depth[depth] as f32,
                self.list_node_single_byte_keys_by_depth[depth],
                self.list_node_single_byte_keys_by_depth[depth] as f32
                    / self.total_list_nodes_by_depth[depth] as f32
                    * 100.0,
            );
        }
    }
    pub fn count_ocupancy<V: Clone + Send + Sync + Unpin>(map: &PathMap<V>) -> Self {
        let mut counters = Counters::new();

        counters.count_node(map.root().unwrap().as_tagged(), 0);

        let mut zipper = map.read_zipper();
        while zipper.to_next_step() {
            let depth = zipper.path().len();

            counters.run_counter_update(depth);
            if let Some(focus_node) = zipper.get_focus().try_as_tagged() {
                counters.count_node(focus_node, depth);
            } else {
                counters.end_run(depth - 1);
            }
        }

        counters
    }
    fn count_node<V: Clone + Send + Sync, A: crate::alloc::Allocator>(
        &mut self,
        node: TaggedNodeRef<V, A>,
        depth: usize,
    ) {
        if let Some(dbn) = node.as_dense() {
            if dbn.item_count() != 1 {
                self.end_run(depth);
            }
            self.increment_common_counters(node, depth);
            self.total_dense_byte_nodes_by_depth[depth] += 1;
        }
        if let Some(lln) = node.as_list() {
            if lln.item_count() != 1 {
                self.end_run(depth);
            }
            self.increment_common_counters(node, depth);
            self.total_list_nodes_by_depth[depth] += 1;

            let (key0, key1) = lln.get_both_keys();
            self.total_slot0_length_by_depth[depth] += key0.len();
            if key1.len() > 0 {
                self.slot1_occupancy_count_by_depth[depth] += 1;
                self.total_slot1_length_by_depth[depth] += key1.len();
            }
            if key0.len() == 1 || key1.len() == 1 {
                self.list_node_single_byte_keys_by_depth[depth] += 1;
            }
        }
    }
    fn resize_all_historgrams(&mut self, depth: usize) {
        if self.total_nodes_by_depth.len() <= depth {
            self.total_nodes_by_depth.resize(depth + 1, 0);
            self.total_child_items_by_depth.resize(depth + 1, 0);
            self.max_child_items_by_depth.resize(depth + 1, 0);
            self.total_dense_byte_nodes_by_depth.resize(depth + 1, 0);
            self.total_list_nodes_by_depth.resize(depth + 1, 0);
            self.total_slot0_length_by_depth.resize(depth + 1, 0);
            self.slot1_occupancy_count_by_depth.resize(depth + 1, 0);
            self.total_slot1_length_by_depth.resize(depth + 1, 0);
            self.list_node_single_byte_keys_by_depth
                .resize(depth + 1, 0);
        }
    }
    fn increment_common_counters<V: Clone + Send + Sync, A: crate::alloc::Allocator>(
        &mut self,
        node: TaggedNodeRef<V, A>,
        depth: usize,
    ) {
        self.resize_all_historgrams(depth);
        let child_item_count = node.item_count();
        self.total_nodes_by_depth[depth] += 1;
        self.total_child_items_by_depth[depth] += child_item_count;
        if self.max_child_items_by_depth[depth] < child_item_count {
            self.max_child_items_by_depth[depth] = child_item_count;
        }
    }
    fn end_run(&mut self, depth: usize) {
        if depth > self.cur_run_start_depth {
            let cur_run_length = depth - self.cur_run_start_depth;
            self.push_run(cur_run_length, depth - 1);
        }
        self.cur_run_start_depth = depth;
    }
    fn run_counter_update(&mut self, depth: usize) {
        if self.cur_run_start_depth > depth {
            self.cur_run_start_depth = depth;
        }
    }
    fn push_run(&mut self, cur_run_length: usize, byte_depth: usize) {
        if self.run_length_histogram_by_ending_byte_depth.len() <= cur_run_length {
            self.run_length_histogram_by_ending_byte_depth
                .resize(cur_run_length + 1, vec![]);
        }
        if self.run_length_histogram_by_ending_byte_depth[cur_run_length].len() <= byte_depth {
            self.run_length_histogram_by_ending_byte_depth[cur_run_length]
                .resize(byte_depth + 1, 0);
        }
        self.run_length_histogram_by_ending_byte_depth[cur_run_length][byte_depth] += 1;
    }
}

pub fn print_traversal<'a, V: 'a + Clone + Unpin, Z: ZipperIteration + Clone>(zipper: &Z) {
    let mut zipper = zipper.clone();

    println!("{:?}", zipper.path());
    while zipper.to_next_val() {
        println!("{:?}", zipper.path());
    }
}
