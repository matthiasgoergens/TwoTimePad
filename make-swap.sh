#!/bin/bash
fallocate -l 32G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo /swapfile swap swap defaults 0 0 | tee --append /etc/fstab
