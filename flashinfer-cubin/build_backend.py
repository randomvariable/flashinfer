"""
Custom build backend that downloads cubins before building the package.
"""

import os
import sys
from pathlib import Path
from setuptools import build_meta as _orig

# Add parent directory to path to import artifacts module
sys.path.insert(0, str(Path(__file__).parent.parent))

from build_utils import get_git_version

# Skip version check when building flashinfer-cubin package
os.environ["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"


def _download_cubins():
    """Download cubins into the package, reusing the persistent cache."""
    import shutil

    package_cubin_dir = Path(__file__).parent / "flashinfer_cubin" / "cubins"
    cache_cubin_dir = Path(
        os.environ.get(
            "FLASHINFER_CUBIN_BUILD_CACHE", "/root/.cache/flashinfer/cubins"
        )
    )

    package_cubin_dir.mkdir(parents=True, exist_ok=True)
    cache_cubin_dir.mkdir(parents=True, exist_ok=True)

    original_cubin_dir = os.environ.get("FLASHINFER_CUBIN_DIR")
    os.environ["FLASHINFER_CUBIN_DIR"] = str(cache_cubin_dir)

    try:
        from flashinfer.artifacts import download_artifacts

        print(f"Downloading cubins to cache {cache_cubin_dir}...")
        download_artifacts()
        print(f"Successfully updated cubin cache {cache_cubin_dir}")
        if package_cubin_dir.exists():
            shutil.rmtree(package_cubin_dir)
        shutil.copytree(cache_cubin_dir, package_cubin_dir)
        cubin_files = list(package_cubin_dir.rglob("*.cubin"))
        print(f"Packaged {len(cubin_files)} cubin files")
    finally:
        if original_cubin_dir:
            os.environ["FLASHINFER_CUBIN_DIR"] = original_cubin_dir
        else:
            os.environ.pop("FLASHINFER_CUBIN_DIR", None)


def _create_build_metadata():
    """Create build metadata file with version information."""
    version_file = Path(__file__).parent.parent / "version.txt"
    if version_file.exists():
        with open(version_file, "r") as f:
            version = f.read().strip()
    else:
        version = "0.0.0+unknown"

    # Add dev suffix if specified
    dev_suffix = os.environ.get("FLASHINFER_DEV_RELEASE_SUFFIX", "")
    if dev_suffix:
        version = f"{version}.dev{dev_suffix}"

    # Get git version
    git_version = get_git_version(cwd=Path(__file__).parent.parent)

    # Append local version suffix if available
    local_version = os.environ.get("FLASHINFER_LOCAL_VERSION")
    if local_version:
        # Use + to create a local version identifier that will appear in wheel name
        version = f"{version}+{local_version}"

    # Create build metadata in the source tree
    package_dir = Path(__file__).parent / "flashinfer_cubin"
    build_meta_file = package_dir / "_build_meta.py"

    # Check if we're in a git repository
    git_dir = Path(__file__).parent.parent / ".git"
    in_git_repo = git_dir.exists()

    # If file exists and not in git repo (installing from sdist), keep existing file
    if build_meta_file.exists() and not in_git_repo:
        print("Build metadata file already exists (not in git repo), keeping it")
        return version

    # In git repo (editable) or file doesn't exist, create/update it
    with open(build_meta_file, "w") as f:
        f.write('"""Build metadata for flashinfer-cubin package."""\n')
        f.write(f'__version__ = "{version}"\n')
        f.write(f'__git_version__ = "{git_version}"\n')

    print(f"Created build metadata file with version {version}")
    return version


# Create build metadata as soon as this module is imported
_create_build_metadata()


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    """Build a wheel, downloading cubins first."""
    _download_cubins()
    return _orig.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    """Build an editable install, downloading cubins first."""
    _download_cubins()
    return _orig.build_editable(wheel_directory, config_settings, metadata_directory)


# Pass through all other hooks
get_requires_for_build_wheel = _orig.get_requires_for_build_wheel
get_requires_for_build_editable = _orig.get_requires_for_build_editable
prepare_metadata_for_build_wheel = _orig.prepare_metadata_for_build_wheel
prepare_metadata_for_build_editable = _orig.prepare_metadata_for_build_editable
