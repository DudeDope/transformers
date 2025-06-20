# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING

from ...utils import OptionalDependencyNotAvailable, _LazyModule, is_torch_available


_import_structure = {
    "configuration_smolvlm": [
        "SmolVLMConfig",
        "SmolVLMVisionConfig",
    ],
    "processing_smolvlm": ["SmolVLMProcessor"],
}

# Add Swin configuration imports
_import_structure["configuration_smolvlm_swin"] = [
    "SmolVLMSwinConfig",
    "SmolVLMSwinVisionConfig",
]

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["image_processing_smolvlm"] = ["SmolVLMImageProcessor"]
    _import_structure["modeling_smolvlm"] = [
        "SmolVLMForConditionalGeneration",
        "SmolVLMModel",
        "SmolVLMPreTrainedModel",
        "SmolVLMVisionTransformer",
    ]
    
    # Add Swin modeling imports
    _import_structure["modeling_smolvlm_swin"] = [
        "SmolVLMSwinForConditionalGeneration",
        "SmolVLMSwinModel",
        "SmolVLMSwinVisionTransformer",
    ]

if TYPE_CHECKING:
    from .configuration_smolvlm import SmolVLMConfig, SmolVLMVisionConfig
    from .configuration_smolvlm_swin import SmolVLMSwinConfig, SmolVLMSwinVisionConfig
    from .processing_smolvlm import SmolVLMProcessor

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .image_processing_smolvlm import SmolVLMImageProcessor
        from .modeling_smolvlm import (
            SmolVLMForConditionalGeneration,
            SmolVLMModel,
            SmolVLMPreTrainedModel,
            SmolVLMVisionTransformer,
        )
        from .modeling_smolvlm_swin import (
            SmolVLMSwinForConditionalGeneration,
            SmolVLMSwinModel,
            SmolVLMSwinVisionTransformer,
        )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
