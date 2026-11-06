// The tree half of ai/search/batched_mcts.py, in C++: N independent PUCT searches advanced in
// lockstep, one leaf per tree per round, with the network left to Python.
//
// Why: profiled on the real workloads (research/log/search_performance_profile.md), a search
// spent ~90% of its time in Python bookkeeping -- PUCT over ~19 children per node, cloning and
// stepping new children, featurising leaves one at a time -- and ~7% in the network, at 6% GPU
// utilisation. The network still has to run in Python (torch, the model, its weights), so the
// split is: this class does everything between two forward passes, and a round costs Python two
// calls regardless of how many trees there are or how deep they go.
//
//     reset(roots, simulations, seed)
//     k = select_leaves()            # first call: the roots; then one new leaf per tree
//     while k: probs, values = net(obs[:k], masks[:k]); expand_and_backup(probs, values); k = select_leaves()
//
// The search is the Python searcher's, bit for bit -- the same PUCT expression in the same order,
// the same settle depths, values kept from the US side, a node with an empty mask treated as a
// draw, and the same random numbers: each new child's chance seed is `random.Random.getrandbits(64)`
// from the caller's own generator, whose MT19937 state is handed in by reset() and handed back by
// mt_state(), in the order the Python tree draws them (round by round, trees in index order). A
// game searched with either tree is therefore the same game.
//
// Trees are independent, so their descents, featurisation and backups run across the OpenMP team
// (gomp_parallel_for, the pool shared with torch). Only the seeds are handed out serially: a round
// first walks every tree to where it needs a new child, then numbers those trees' seeds in tree
// order, then builds the children. Leaf rows are numbered in tree order too, so neither the seeds
// nor the batch handed to the network depend on the thread count.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "ts/action_mask.hpp"
#include "ts/cached_state.hpp"
#include "ts/engine.hpp"
#include "ts/game_state.hpp"
#include "ts/invariant.hpp"
#include "ts/micro_action.hpp"
#include "ts/observation.hpp"
#include "gomp_parallel.hpp"

namespace ts_search {

struct Edge {
    uint16_t action;
    double prior;
    double n;
    double w;      // sum of backed-up values, US side
    int32_t child; // node index, -1 until first taken
};

struct Node {
    // The position with its own legal mask (ts::CachedState): a child is validated against its
    // parent's, and the mask auto-advance generated where it stopped is the one handed to the
    // network -- one generation per node where there were three.
    ts::CachedState state;
    int8_t mover = 0;          // ts::Player of the side to move, 0 at a terminal
    bool terminal = false;
    bool expanded = false;     // priors filled in by a network evaluation
    double value_us = 0.0;
    double total = 0.0;        // sum of the edges' n
    uint32_t edge_begin = 0;
    uint32_t edge_count = 0;
};

// CPython's random.Random: MT19937 with the state layout of getstate()/setstate() -- 624 words and
// the index of the next one -- and getrandbits(64) as two 32-bit outputs, least significant first.
struct PyRandom {
    static constexpr int N = 624;
    static constexpr int M = 397;
    uint32_t mt[N] = {};
    int index = N;

    uint32_t next32() {
        static constexpr uint32_t mag01[2] = {0x0U, 0x9908b0dfU};
        if (index >= N) {
            int kk = 0;
            uint32_t y;
            for (; kk < N - M; ++kk) {
                y = (mt[kk] & 0x80000000U) | (mt[kk + 1] & 0x7fffffffU);
                mt[kk] = mt[kk + M] ^ (y >> 1) ^ mag01[y & 0x1U];
            }
            for (; kk < N - 1; ++kk) {
                y = (mt[kk] & 0x80000000U) | (mt[kk + 1] & 0x7fffffffU);
                mt[kk] = mt[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 0x1U];
            }
            y = (mt[N - 1] & 0x80000000U) | (mt[0] & 0x7fffffffU);
            mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ mag01[y & 0x1U];
            index = 0;
        }
        uint32_t y = mt[index++];
        y ^= (y >> 11);
        y ^= (y << 7) & 0x9d2c5680U;
        y ^= (y << 15) & 0xefc60000U;
        y ^= (y >> 18);
        return y;
    }
    uint64_t getrandbits64() {
        const uint64_t lo = next32();
        const uint64_t hi = next32();
        return (hi << 32) | lo;
    }
    void set_state(const std::vector<uint64_t>& st) {
        if (st.size() != static_cast<size_t>(N) + 1 || st[N] > static_cast<uint64_t>(N))
            throw std::invalid_argument("BatchedSearch: expected random.Random's 624 words and index");
        for (int i = 0; i < N; ++i) mt[i] = static_cast<uint32_t>(st[static_cast<size_t>(i)]);
        index = static_cast<int>(st[N]);
    }
    std::vector<uint64_t> state() const {
        std::vector<uint64_t> st(static_cast<size_t>(N) + 1);
        for (int i = 0; i < N; ++i) st[static_cast<size_t>(i)] = mt[i];
        st[N] = static_cast<uint64_t>(index);
        return st;
    }
};

struct Tree {
    std::vector<Node> nodes;   // nodes[0] is the root
    std::vector<Edge> edges;
    std::vector<std::pair<int32_t, int32_t>> path;  // (node, edge) of the pending descent
    int32_t new_edge = -1;     // this round's descent stopped at an untaken edge
    uint64_t seed = 0;         // the chance seed of the child it will create
    int64_t budget = 0;        // this search's simulations (reset)
    int64_t remaining = 0;
    int32_t leaf = -1;         // node awaiting evaluation, -1 if none
    int32_t row = -1;          // its row in the leaf buffers
};

inline ts::Player acting_player(const ts::GameState& s) {
    const ts::Player d = s.ctx().decision_player;
    return d != ts::Player::NONE ? d : s.phasing_player;
}

class BatchedSearch {
public:
    BatchedSearch(size_t capacity, double c_puct, bool auto_advance, uint32_t obs_features,
                  bool merged_influence, double fpu_reduction = 0.0)
        : capacity_(capacity), c_puct_(c_puct), fpu_(fpu_reduction), auto_advance_(auto_advance),
          features_(obs_features), merged_(merged_influence),
          obs_width_(ts::OBS_SIZE_V23 + ts::obs_features::extra_width(obs_features)),
          own_obs_(capacity * obs_width_, 0.0f), own_masks_(capacity * ts::FLAT_ACTION_SPACE_SIZE, 0),
          obs_(own_obs_.data()), masks_(own_masks_.data()) {
        trees_.reserve(capacity);
        if (obs_features & ~ts::obs_features::ALL)
            throw std::invalid_argument("BatchedSearch: unknown observation feature bits");
    }

    size_t capacity() const { return capacity_; }
    size_t obs_width() const { return obs_width_; }
    size_t num_trees() const { return trees_.size(); }
    size_t num_groups() const { return groups_.size(); }
    float* obs_data() { return obs_; }
    uint8_t* mask_data() { return masks_; }

    // Write leaves into caller-owned buffers instead of this object's own -- page-locked memory, so
    // the copy to the GPU runs asynchronously. The caller keeps them alive while this object uses
    // them; capacity x obs_width floats and capacity x FLAT_ACTION_SPACE_SIZE bytes.
    // This object's own buffers are freed then: at capacity 8,192 they are half a gigabyte.
    void set_buffers(float* obs, uint8_t* masks) {
        obs_ = obs;
        masks_ = masks;
        std::vector<float>().swap(own_obs_);
        std::vector<uint8_t>().swap(own_masks_);
    }

    // Start one search per root, the trees split into `num_groups` contiguous groups of (nearly)
    // equal size. Each group's leaves are handed out and evaluated on their own, so one group's
    // tree work can run while the network evaluates another's. The grouping is part of what the
    // search computes, not only how fast: the network's output for a row depends on the batch it
    // is evaluated in, so the Python tree groups the same way (ai/search/batched_mcts.py).
    // `simulations` is each tree's own budget; `rng_state` is the caller's random.Random state
    // (getstate()[1]), which the children's chance seeds are drawn from.
    void reset(const std::vector<ts::GameState>& roots, int64_t simulations,
               const std::vector<uint64_t>& rng_state, size_t num_groups) {
        reset(roots, std::vector<int64_t>(roots.size(), simulations), rng_state, num_groups);
    }

    // The same with a budget per tree, so searches of different sizes share one set of rounds
    // (the Gumbel root's candidates, ai/search/gumbel_root.py). A tree stops taking rounds when its
    // own budget is spent; the rounds still visit trees in index order, so the seeds come out as
    // the Python tree draws them with the same budgets.
    void reset(const std::vector<ts::GameState>& roots, const std::vector<int64_t>& simulations,
               const std::vector<uint64_t>& rng_state, size_t num_groups) {
        if (simulations.size() != roots.size())
            throw std::invalid_argument("BatchedSearch: one budget per root, got " +
                                        std::to_string(simulations.size()) + " for " +
                                        std::to_string(roots.size()) + " roots");
        rng_.set_state(rng_state);
        if (roots.size() > capacity_)
            throw std::invalid_argument("BatchedSearch: " + std::to_string(roots.size()) +
                                        " roots exceed capacity " + std::to_string(capacity_));
        if (num_groups == 0 || (num_groups > roots.size() && !roots.empty()))
            throw std::invalid_argument("BatchedSearch: need 1 <= num_groups <= number of roots");
        trees_.resize(roots.size());
        for (size_t i = 0; i < roots.size(); ++i) {
            Tree& t = trees_[i];
            t.nodes.clear();
            t.edges.clear();
            t.path.clear();
            // Each round adds at most one node, so this never reallocates during the search, and
            // the vectors keep their capacity from one reset to the next -- unless it is more than
            // twice what this search can use. A node holds a whole GameState (1.5 KB), so a tree
            // that once searched 256 simulations and now searches 10 would otherwise keep a
            // third of a megabyte it never touches again.
            t.budget = std::max<int64_t>(0, simulations[i]);
            const size_t need = static_cast<size_t>(t.budget) + 1;
            if (t.nodes.capacity() > 2 * need) {
                std::vector<Node>().swap(t.nodes);
                std::vector<Edge>().swap(t.edges);
            }
            t.nodes.reserve(need);
            // The network models both sides, so the tree plays both under its rule bits.
            ts::GameState root = roots[i];
            ts::seat_both(root, features_);
            t.nodes.push_back(make_node(root));
            t.remaining = 0;
            t.leaf = -1;
            t.row = -1;
        }
        groups_.clear();
        const size_t n = roots.size();
        const size_t per = n == 0 ? 0 : (n + num_groups - 1) / num_groups;
        for (size_t b = 0; b < n || groups_.empty(); b += per) {
            groups_.push_back({b, std::min(n, b + per)});
            if (per == 0) break;
        }
        roots_pending_.assign(groups_.size(), true);
        roots_phase_ = true;
    }

    // First row of group g's leaves in the buffers: its leaves fill [group_offset(g), +k).
    size_t group_offset(size_t g) const { return group(g).first; }

    // Group g's leaves awaiting evaluation, written into rows [group_offset(g), +k); returns k.
    // While the roots are being evaluated (the first call per group after reset) it returns the
    // group's roots. After that each call is one simulation round for the group: every tree still
    // holding budget walks to a leaf, terminal leaves are backed up at once, and the rest are the
    // rows returned -- possibly none. Rounds must be taken group 0, 1, ... in turn (as the Python
    // tree takes them), so that the chance seeds are drawn in its order.
    size_t select_leaves(size_t g) {
        const auto [b, e] = group(g);
        size_t k = 0;
        if (roots_phase_) {
            if (!roots_pending_[g])
                throw std::logic_error("select_leaves: this group's roots were already evaluated");
            for (size_t i = b; i < e; ++i) {
                Tree& t = trees_[i];
                if (!t.nodes[0].terminal && !t.nodes[0].expanded) {
                    t.leaf = 0;
                    t.path.clear();
                    t.row = static_cast<int32_t>(b + k++);
                }
            }
            if (k > 0) {
                featurise_rows(b, e);
            } else {
                roots_done(g);
            }
            return k;
        }
        for_each_tree(b, e, [this](Tree& t) {
            t.leaf = -1;
            t.new_edge = -1;
            if (t.remaining <= 0) return;
            t.remaining -= 1;
            walk(t);
        });
        for (size_t i = b; i < e; ++i)
            if (trees_[i].new_edge >= 0) trees_[i].seed = rng_.getrandbits64();
        for_each_tree(b, e, [this](Tree& t) {
            if (t.new_edge >= 0) create_child(t);
            if (t.leaf < 0) return;
            const Node& leaf = t.nodes[static_cast<size_t>(t.leaf)];
            if (leaf.terminal) {
                backup(t, leaf.value_us);
                t.leaf = -1;
            }
        });
        for (size_t i = b; i < e; ++i)
            if (trees_[i].leaf >= 0) trees_[i].row = static_cast<int32_t>(b + k++);
        if (k > 0) featurise_rows(b, e);
        return k;
    }

    // Fill in group g's pending leaves from the network: `probs` is (k, FLAT_ACTION_SPACE_SIZE),
    // the softmax over the masked logits, and `values` is (k,), v_win from each leaf's mover's
    // side -- row j being the leaf in buffer row group_offset(g) + j.
    void expand_and_backup(size_t g, const float* probs, const float* values, size_t k) {
        const auto [b, e] = group(g);
        for (size_t i = b; i < e; ++i) {
            const Tree& t = trees_[i];
            if (t.leaf >= 0 && (t.row < static_cast<int32_t>(b) || static_cast<size_t>(t.row) >= b + k))
                throw std::invalid_argument("expand_and_backup: fewer rows than pending leaves");
        }
        const bool roots = roots_phase_;
        for_each_tree(b, e, [&](Tree& t) {
            if (t.leaf < 0) return;
            const size_t r = static_cast<size_t>(t.row) - b;
            expand(t, static_cast<size_t>(t.leaf), probs + r * ts::FLAT_ACTION_SPACE_SIZE,
                   masks_ + static_cast<size_t>(t.row) * ts::FLAT_ACTION_SPACE_SIZE, values[r]);
            if (!roots) backup(t, t.nodes[static_cast<size_t>(t.leaf)].value_us);
            t.leaf = -1;
            t.row = -1;
        });
        if (roots) roots_done(g);
    }

    // Is any tree of group g still holding budget? (False while the roots are being evaluated.)
    bool group_has_budget(size_t g) const {
        if (roots_phase_) return false;
        const auto [b, e] = group(g);
        for (size_t i = b; i < e; ++i)
            if (trees_[i].remaining > 0) return true;
        return false;
    }

    bool roots_phase() const { return roots_phase_; }

    // Replace a root's priors (root noise is drawn by the caller, from its own stream).
    void set_root_priors(size_t i, const std::vector<double>& priors) {
        Tree& t = tree(i);
        const Node& root = t.nodes[0];
        if (priors.size() != root.edge_count)
            throw std::invalid_argument("set_root_priors: one prior per root action");
        for (uint32_t e = 0; e < root.edge_count; ++e) t.edges[root.edge_begin + e].prior = priors[e];
    }

    // The caller's random.Random state after the draws this search made (for setstate()).
    std::vector<uint64_t> mt_state() const { return rng_.state(); }

    const Node& root(size_t i) { return tree(i).nodes[0]; }
    const Edge& root_edge(size_t i, size_t e) {
        Tree& t = tree(i);
        return t.edges[t.nodes[0].edge_begin + e];
    }
    size_t tree_size(size_t i) { return tree(i).nodes.size(); }

private:
    const std::pair<size_t, size_t>& group(size_t g) const {
        if (g >= groups_.size()) throw std::out_of_range("BatchedSearch: no group " + std::to_string(g));
        return groups_[g];
    }

    // Group g's roots are evaluated; the simulations start once every group's are.
    void roots_done(size_t g) {
        roots_pending_[g] = false;
        for (bool p : roots_pending_)
            if (p) return;
        begin_simulations();
    }

    Tree& tree(size_t i) {
        if (i >= trees_.size()) throw std::out_of_range("BatchedSearch: no tree " + std::to_string(i));
        return trees_[i];
    }

    static Node make_node(const ts::GameState& s) {
        Node nd;
        nd.state = ts::CachedState(s);
        set_node_kind(nd);
        return nd;
    }

    // Terminal (with its utility) or the side to move, from the node's state.
    static void set_node_kind(Node& nd) {
        const ts::GameState& s = nd.state.state();
        if (ts::Engine::is_terminal(s)) {
            nd.terminal = true;
            nd.expanded = true;
            nd.value_us = static_cast<double>(ts::Engine::get_terminal_utility(s));
        } else {
            nd.mover = static_cast<int8_t>(acting_player(s));
        }
    }

    void begin_simulations() {
        roots_phase_ = false;
        for (Tree& t : trees_) {
            const Node& r = t.nodes[0];
            t.remaining = (!r.terminal && r.edge_count > 0)
                ? std::max<int64_t>(0, t.budget - static_cast<int64_t>(r.total)) : 0;
        }
    }

    // ai/search/batched_mcts.py `settle`: FORCED is the engine's own auto-advance, CHANCE drains
    // die rolls only.
    void settle(ts::CachedState& s) const {
        if (auto_advance_) {
            s.auto_advance();
            return;
        }
        while (!ts::Engine::is_terminal(s.state()) && s.state().ctx().decision_player == ts::Player::NONE &&
               s.state().ctx().decision_type == ts::DecisionType::ROLL_DIE) {
            if (!s.step(ts::MicroAction{ts::DecisionType::ROLL_DIE, 0, 0, 0})) break;
        }
    }

    // PUCT, read from the mover's side of a US-side value. The expression and its evaluation
    // order are the Python searcher's, so the choice is the same to the last bit.
    int32_t select(const Tree& t, const Node& nd) const {
        const double total = nd.total;
        const double sqrt_total = std::sqrt(total > 1.0 ? total : 1.0);
        const bool us_moves = nd.mover == static_cast<int8_t>(ts::Player::US);
        int32_t best_i = 0;
        double best_v = -1e30;
        for (uint32_t i = 0; i < nd.edge_count; ++i) {
            const Edge& e = t.edges[nd.edge_begin + i];
            double q = e.n > 0 ? e.w / e.n : nd.value_us;
            if (!us_moves) q = -q;
            // First-play urgency: an unvisited move is valued at the node's own value, less fpu_
            // (from the mover's side). 0 is the original rule.
            if (e.n == 0) q -= fpu_;
            const double v = q + c_puct_ * e.prior * sqrt_total / (1.0 + e.n);
            if (v > best_v) {
                best_v = v;
                best_i = static_cast<int32_t>(i);
            }
        }
        return best_i;
    }

    // Walk down by PUCT. Stops at a terminal or unexpanded node (the leaf), or at an edge never
    // taken, whose child `create_child` builds once the round's seeds are handed out.
    void walk(Tree& t) const {
        t.path.clear();
        size_t node = 0;
        while (true) {
            const Node& nd = t.nodes[node];
            if (nd.terminal || !nd.expanded) {
                t.leaf = static_cast<int32_t>(node);
                return;
            }
            const int32_t idx = select(t, nd);
            const uint32_t edge = nd.edge_begin + static_cast<uint32_t>(idx);
            t.path.emplace_back(static_cast<int32_t>(node), static_cast<int32_t>(edge));
            const int32_t child = t.edges[edge].child;
            if (child < 0) {
                t.new_edge = static_cast<int32_t>(edge);
                return;
            }
            node = static_cast<size_t>(child);
        }
    }

    void create_child(Tree& t) {
        const size_t parent = static_cast<size_t>(t.path.back().first);
        Edge& e = t.edges[static_cast<size_t>(t.new_edge)];
        // Built in its own slot, from a copy of the parent -- one copy of the state where there were
        // three. reset() reserved a node for every simulation, so the vector does not move and
        // `parent` and `e` stay valid. The copy carries the parent's legal mask, and the action
        // is validated against it: only the chance seed differs, and the mask does not read it.
        if (t.nodes.size() == t.nodes.capacity())
            throw std::logic_error("BatchedSearch: a tree outgrew the nodes reserved for it");
        t.nodes.push_back(t.nodes[parent]);
        Node& child = t.nodes.back();
        child.state.set_rng(t.seed);
        if (!child.state.step_flat(e.action, merged_)) {
            throw std::runtime_error("BatchedSearch: the engine refused flat action " +
                                     std::to_string(e.action) + " taken from its own mask");
        }
        settle(child.state);
        child.mover = 0;
        child.terminal = false;
        child.expanded = false;
        child.value_us = 0.0;
        child.total = 0.0;
        child.edge_begin = 0;
        child.edge_count = 0;
        set_node_kind(child);
        e.child = static_cast<int32_t>(t.nodes.size() - 1);
        t.leaf = e.child;
        t.new_edge = -1;
    }

    static void backup(Tree& t, double value_us) {
        for (const auto& [node, edge] : t.path) {
            Edge& e = t.edges[static_cast<size_t>(edge)];
            e.n += 1.0;
            e.w += value_us;
            t.nodes[static_cast<size_t>(node)].total += 1.0;
        }
    }

    void expand(Tree& t, size_t node, const float* probs, const uint8_t* mask, float value) {
        Node& nd = t.nodes[node];
        uint32_t k = 0;
        double total = 0.0;
        for (size_t a = 0; a < ts::FLAT_ACTION_SPACE_SIZE; ++a) {
            if (mask[a]) {
                ++k;
                total += static_cast<double>(probs[a]);
            }
        }
        if (k == 0) {
            // A live position with nothing legal is an engine defect, not a draw: logged loudly,
            // as the runner logs it, and then valued 0 so the search can finish.
            ts::report_anomaly("search expanded a non-terminal node with an empty mask",
                               nd.state.state());
            nd.terminal = true;
            nd.value_us = 0.0;
            nd.expanded = true;
            return;
        }
        nd.edge_begin = static_cast<uint32_t>(t.edges.size());
        nd.edge_count = k;
        const bool uniform = !(total > 1e-12);
        for (size_t a = 0; a < ts::FLAT_ACTION_SPACE_SIZE; ++a) {
            if (!mask[a]) continue;
            const double p = uniform ? 1.0 / static_cast<double>(k)
                                     : static_cast<double>(probs[a]) / total;
            t.edges.push_back(Edge{static_cast<uint16_t>(a), p, 0.0, 0.0, -1});
        }
        nd.total = 0.0;
        const double v = static_cast<double>(value);
        nd.value_us = nd.mover == static_cast<int8_t>(ts::Player::US) ? v : -v;
        nd.expanded = true;
    }

    // `body(tree)` for every tree across the OpenMP team. An exception cannot cross the team, so
    // the first one is kept and rethrown here once every thread is done.
    template <class F>
    void for_each_tree(size_t b, size_t e, F&& body) {
        std::exception_ptr err;
        std::mutex mu;
        gomp_parallel_for(static_cast<int64_t>(e - b), [&](int64_t i) {
            try {
                body(trees_[b + static_cast<size_t>(i)]);
            } catch (...) {
                std::lock_guard<std::mutex> lock(mu);
                if (!err) err = std::current_exception();
            }
        });
        if (err) std::rethrow_exception(err);
    }

    void featurise_rows(size_t b, size_t e) {
        for_each_tree(b, e, [this](Tree& t) {
            if (t.leaf < 0) return;
            Node& nd = t.nodes[static_cast<size_t>(t.leaf)];
            const ts::GameState& s = nd.state.state();
            const size_t r = static_cast<size_t>(t.row);
            float* row = obs_ + r * obs_width_;
            const size_t w = ts::extract_observation_features(s, acting_player(s), features_, row);
            if (w < obs_width_) std::memset(row + w, 0, (obs_width_ - w) * sizeof(float));
            uint8_t* mask_row = masks_ + r * ts::FLAT_ACTION_SPACE_SIZE;
            if (merged_) {
                ts::Engine::get_flat_action_mask(s, mask_row, true);   // another view: not cached
            } else {
                std::memcpy(mask_row, nd.state.legal_mask(), ts::FLAT_ACTION_SPACE_SIZE);
            }
        });
    }

    size_t capacity_;
    double c_puct_;
    double fpu_;
    bool auto_advance_;
    uint32_t features_;
    bool merged_;
    size_t obs_width_;
    std::vector<float> own_obs_;
    std::vector<uint8_t> own_masks_;
    float* obs_;              // own_obs_, or the caller's page-locked buffer (set_buffers)
    uint8_t* masks_;
    std::vector<Tree> trees_;
    PyRandom rng_;
    bool roots_phase_ = false;
    std::vector<std::pair<size_t, size_t>> groups_;  // [begin, end) of each group's trees
    std::vector<bool> roots_pending_;
};

}  // namespace ts_search
