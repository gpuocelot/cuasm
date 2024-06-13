# Package for ArchLinux

This repository contains a `PKGBUILD` file to install this software as a package into your ArchLinux system.


## Setup

1. Make sure your cwd is in the directory where PKGBUILD is located, which is the root directory of this repo:

```
cd /path/that/contains/PKGBUILD
```

2. Prepare the package and install it:

```
makepkg -si

```

or:

```
makepkg -s
sudo pacman -U python-cuasm-git-0.1-1-x86_64.pkg.tar.zst
```


## Usage

Executables `cuasm`, `dsass`, `hnvcc` and `hcubin` are available in `/usr/bin`.

