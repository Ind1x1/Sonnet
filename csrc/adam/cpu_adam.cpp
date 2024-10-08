// Copyright (c) Microsoft Corporation.
// SPDX-License-Identifier: Apache-2.0

// DeepSpeed Team

#include "cpu_adam.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("adam_update", &ds_adam_step, "DeepSpeed CPU Adam update (C++)");
    m.def("adam_update_copy",
          &ds_adam_step_plus_copy,
          "DeepSpeed CPU Adam update and param copy (C++)");
    m.def("create_adam", &create_adam_optimizer, "DeepSpeed CPU Adam (C++)");
    m.def("destroy_adam", &destroy_adam_optimizer, "DeepSpeed CPU Adam destroy (C++)");
    
    m.def("create_sonnet_adam", &create_sonnet_adam_optimizer, "Sonnet CPU Adam (C++)");
    m.def("destroy_sonnet_adam", &destroy_sonnet_adam_optimizer, "Sonnet CPU Adamdestroy (C++)");
    m.def("sonnet_update", &sonnet_adam_step, "Sonnet CPU Adam update (C++)");
}
