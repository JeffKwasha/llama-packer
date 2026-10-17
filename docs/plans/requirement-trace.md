# Requirement → test trace

Every rule the design states, and the test that holds it up. This exists
because the reverse order was used once and it cost real time: `met_demands`
was deleted on a call-graph reading, and the `require` gate went with it —
a text-only model could then claim `require: [vision]`, an id promising a
capability the entry does not advertise.

The rule that prevents that: **design → docs → tests → code.** A requirement
is not implemented until a test named here fails without the code. Code is
never removed on a call-graph reading alone.

Run `python extras/check_trace.py` to verify every test named below exists
and every one of them carries a `Validates:` citation.

## The category is a declaration

| # | Requirement | Test |
|---|---|---|
| 1 | The category comes from the sidecar `usage:` tag; nothing infers or overrides it | `test_usage_tags_resolve_to_declared_demands` |
| 2 | An undeclared tag warns and is ignored — only declared categories exist | `test_usage_tag_not_declared_warns_and_is_ignored` |
| 3 | Each satisfiable category emits its own id | `test_every_category_gets_an_entry` |
| 4 | A category that cannot be met at any rung emits **nothing** | `test_rung_is_none_when_the_category_cannot_be_served`, `test_plan_vision_and_text_both_over_budget_disables` |
| 5 | Every model also gets a fallback entry with no category segment | `test_untagged_gets_a_bare_id`, `test_every_model_gets_a_fallback` |
| 6 | The fallback can itself fail and emit nothing | `test_model_with_no_rung_at_all_emits_nothing` |

## Vision and the ladder

| # | Requirement | Test |
|---|---|---|
| 7 | Vision is shed when shedding it buys context | `test_rung_drops_vision_when_it_buys_context`, `test_category_prefers_context_over_capability` |
| 8 | Vision is kept when it buys nothing — no capability discarded for no gain | `test_rung_keeps_vision_when_it_is_free` |
| 9 | A rung that does not fit is skipped, and the next along the order tried | `test_rung_honours_keep_then_falls_back_to_vision` |
| 10 | A **category** takes the rung buying the most context | `test_category_prefers_context_over_capability` |
| 11 | The **fallback** takes the first qualifying rung — capability first | `test_fallback_keeps_capability_until_the_minimum_forces_it` |
| 12 | The fallback sheds only when holding a capability leaves no legal serving | `test_fallback_sheds_when_nothing_else_fits` |
| 13 | `sacrifice:` order drives the ladder, not a global order | `test_shed_levels_are_prefixes_of_the_declared_order` |
| 14 | Capabilities a category omits are shed last, in the `shed:` order | `test_unnamed_capabilities_come_last_not_first`, `test_shed_order_is_configurable` |
| 15 | `require:` is never shed | `test_required_capabilities_are_never_shed` |
| 16 | `require:` gates **existence** — a model lacking it emits nothing | `test_require_gates_existence`, `test_require_prevents_the_entry_emitted` |
| 17 | `keep:` forbids shedding but does **not** gate existence | `test_keep_protects_from_shedding_without_gating_existence` |
| 18 | `keep:` is enforced against a real draft axis | `test_keep_draft_is_enforced_now_that_the_axis_exists` |

## Context floors

| # | Requirement | Test |
|---|---|---|
| 19 | `context_length:` is a ceiling, applied in **both** solve branches | `test_sidecar_ceiling_binds_in_the_squeeze_branch`, `test_sidecar_ceiling_is_a_min_not_a_fixed_cap` |
| 20 | Native vision (no mmproj file) still gets the image floor | `test_calc_ctx_image_floor_native_vision` |
| 21 | Only `--max-context` pins; `context_length:` is never a floor | `test_serving_pin_ignores_the_design_max`, `test_explicit_max_context_still_pins_and_floors`, `test_design_max_is_a_ceiling_not_a_floor` |
| 22 | A declared category outranks the tools heuristic | `test_category_floor_beats_the_tools_heuristic` |
| 23 | `min_context:` and `--max-context` outrank a category | `test_category_floor_never_overrides_the_operator` |

## Image resolution budget

| # | Requirement | Test |
|---|---|---|
| 24 | A megapixel budget converts to tokens through the projector geometry | `test_min_image_megapixels_converts_through_geometry` |
| 25 | The image floor includes the output reserve | `test_min_image_megapixels_floor_includes_output_reserve` |
| 26 | A projector that does not scale with pixels is warned and ignored | `test_min_image_megapixels_ignored_on_non_proportional` |
| 27 | A budget above the declared cap is warned and clamped | `test_min_image_megapixels_too_large_warns` |
| 28 | `min_image_megapixels: 0` drops the requirement and warns | `test_min_image_megapixels_zero_allows_shedding` |
| 29 | A megapixel budget keeps vision from being shed | `test_min_image_megapixels_keeps_vision` |
| 30 | No budget leaves vision shed-able; 0 drops the requirement | `test_no_budget_leaves_vision_shed_able`, `test_zero_budget_drops_the_vision_requirement` |

## Entry ids

| # | Requirement | Test |
|---|---|---|
| 31 | The id carries no context — 182k → 180k must not rename a model | `test_duplicate_ids_do_not_move_when_context_moves` |
| 32 | An untagged entry is the bare stem | `test_untagged_gets_a_bare_id` |
| 33 | A name that cannot be slugged drops the segment, not the entry | `test_unusable_category_name_falls_back_to_a_bare_id` |
| 34 | Ids stay stable across repacks; the disambiguator describes the serving | `test_same_model_two_groups_disambiguated` |

## VRAM safety

| # | Requirement | Test |
|---|---|---|
| 35 | No emitted entry may need more VRAM than exists | `test_calc_ctx_never_reports_a_context_it_cannot_afford`, `test_overallocation_guard_catches_an_injected_violation` |
| 36 | The guard stays quiet and drops nothing on a sane plan | `test_overallocation_guard_is_quiet_when_the_plan_is_sane` |
| 37 | A CPU-resident model gets no free slots | `test_cpu_resident_chat_gets_no_auto_parallel` (planned) — covered today by `test_calc_ctx_cpu_resident_returns_design` |
| 38 | `tools` is demoted on the entry's **own** served context | `test_plan_demotes_tools_below_the_threshold_without_a_matrix`, `test_plan_keeps_tools_when_the_context_clears_the_threshold` |

## Allocation

| # | Requirement | Test |
|---|---|---|
| 39 | Residents serve a fixed small context and are charged for it | `test_resident_rag_is_reserved_at_its_fixed_context` |
| 40 | Resident parallelism is ignored — never buys context from chat | `test_resident_parallelism_is_ignored` |
| 41 | Admission is per entry, on the resolved serving | `test_set_admission_is_per_entry_not_per_stem` |
| 42 | The fallback joins no set | `test_the_fallback_entry_joins_no_set` |
| 43 | The fallback runs the slot solve at exponent 0 | `test_fallback_slot_exponent_is_zero` |
| 44 | Ties in the slot score go to more slots | `test_ties_go_to_more_slots` |

## Deploy

| # | Requirement | Test |
|---|---|---|
| 45 | Installs are atomic; a truncated tree is never certified | `test_cpp_truncated_dir_is_not_certified_as_installed` |
| 46 | The pruner never removes a bundle a symlink points at | `test_prune_never_removes_the_bundle_a_symlink_points_at` |
| 47 | An ambiguous target selector exits non-zero | `test_ambiguous_prefix_is_an_error_not_a_guess` |
| 48 | `cpu`/`cuda` do not silently install Vulkan | `test_cpu_backend_does_not_install_vulkan` |
| 49 | llama-swap validates as the service user, not root | `test_validate_runs_as_the_service_user` |
| 50 | The unit is verified before it becomes the unit systemd reads | `test_rejected_unit_leaves_the_previous_one_installed`, `test_candidate_name_is_a_valid_unit_name` |
| 51 | An unreadable `HF_HOME` warns instead of crashing the installer | `test_report_resolved_warns_when_hf_home_is_unreadable` |
| 52 | The unit is enableable: it carries `[Install] WantedBy=` | `test_unit_is_enableable` |
| 53 | `InaccessiblePaths=` lists only directories present on this host | `test_render_unit_lists_only_present_hidden_dirs`, `test_inaccessible_paths_omits_absent` |
