# components/

用于存放**非安装包形式**（无 PyPI、无稳定 API、仅含脚本与 `nn.Module`）的第三方研究代码库。此类代码不符合 `anomalib` 模型库标准，需作为独立的第三方依赖引入。

每一个目录都是指向**我们自己 Fork 仓库的 Git Submodule**（例如 `components/dinomaly` $\rightarrow$ `aurora0543/Dinomaly`）。

克隆本项目后，请运行以下命令初始化：

```bash
git submodule update --init --recursive
```

#### 添加新的依赖库：

```bash
gh repo fork <upstream-owner>/<repo> --clone=false
git submodule add https://github.com/<you>/<repo>.git components/<name>

```

---

### 开发规范

1. **不可直接修改 `components/<name>**`：所有修复必须提交到 Fork 仓库（在子模块目录下 commit & push），然后更新主仓库的 submodule 指针。`git submodule status` 不应显示未经提交的 `+` 状态。
2. **更新 Submodule 指针**：
```bash
cd components/<name> && git fetch && git checkout <ref>
# 回到主项目提交指针变更

```


3. **结构限制**：一个仓库对应一个子目录（如 `components/dinomaly/`）。
4. **统一隔离导入**：
* **严禁直接 import**：第三方代码的顶层模块名（如 `utils`, `dataset`）必须只在 `core/vendor.py` 中通过 `VendoredRepo` 声明。
* **统一调用**：业务代码一律使用 `import_vendor()["models.uad"].ViTill` 方式调用。CI 脚本（`tests/test_vendor_boundary.py`）会自动检查并拦截违法导入。


5. **适配器翻译**：在 `src/fabric_defect_hub/models/<name>/` 下实现对应的 Adapter，负责在本项目数据结构（`Sample`/`Prediction`/`Artifact`）与第三方代码间进行转换。

---

### 什么时候不应该使用 Submodule？

若第三方代码**无法独立运行**（如依赖外部大型框架，或依赖特定 CUDA 编译扩展如 `mamba_ssm`），应采取**重构/按接口重新实现**的方式（参考 `models/mambaad/`），而非强行 Vendor 化。

---

### 命名空间冲突解决方案（`VendoredRepo`）

不同研究代码常包含同名顶层模块（如 `utils.py`, `dataset.py`），若直接放入 `sys.path` 会导致模块覆盖污染。

本项目通过 `core/vendor.py::VendoredRepo` 动态管理导入生命周期：

* 在导入时临时将目标路径置顶于 `sys.path` 并清理 `sys.modules` 冲突项；
* 完成导入后将模块存入私有缓存并恢复原始全局环境。

**注意**：此机制要求被引入的代码不能在函数内部延迟导入（Lazy Import）其顶层模块。在更新子模块指针时，请务必核验此约束。