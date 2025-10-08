/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * As of early 25.10, legion headers are broken with C++20 which we need.
 * We can hack around that by removing the legion redop half definition
 * so let's just *assume* we can find the correct header here (where it should
 * be).
 */
#include "legate/deps/legion_defines.h"  // if this fails, try removing the hack!

#undef LEGION_REDOP_HALF
