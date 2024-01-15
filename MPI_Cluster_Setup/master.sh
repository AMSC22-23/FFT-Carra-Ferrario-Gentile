#!/bin/bash

#echo "[INFO] checking if mpiuser exists."
if ! sudo grep "mpiuser" /etc/passwd ;then
	echo "[INFO] mpiuser not found."
	sudo useradd -m mpiuser
	echo "[INFO] created new user mpiuser"
fi

sudo su mpiuser -c "mkdir -p shared_storage"

echo "[INFO] setting shared_storage"
exports=$(grep -v "/home/mpiuser/shared_storage *(rw,sync,no_root_squash,no_subtree_check)" /etc/exports)
sudo echo $exports > /etc/exports
sudo echo "/home/mpiuser/shared_storage *(rw,sync,no_root_squash,no_subtree_check)" >> /etc/exports
echo "[INFO] restarting the NFS server"
exportfs -a
sudo systemctl restart nfs-kernel-server

#adding mpiuser to xhosts
xhost si:localuser:mpiuser

echo "[INFO] master configuration complete."

#mpirun -np 2 -host $1,localhost $2
