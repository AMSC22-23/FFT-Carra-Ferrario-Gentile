# MPI Cluster Configuration
This guide provides instructions for setting up a simple MPI cluster for testing purposes. In this setup, one computer acts as the master node, while the others act as worker nodes, simulating an HPC cluster.

## Requirements
Before starting, ensure that all nodes in the cluster have the **same version of MPI installed** and `nfs-server` is installed on the *master node*. Depending on your Linux distribution, you can install it using one of the following commands:

Debian-based:
```bash
sudo apt install nfs-kernel-server
```
Arch-based:
```bash
pacman -Sy nfs-utils
```

## Master node setup
On the master node, execute the `master.sh` script:
```bash
sudo ./master.sh
```

This script creates a new user named **mpiuser** and then configures and exposes a shared NFS storage in `/home/mpiuser/shared_storage`.

## Worker node setup
On the master node, execute the `node.sh` script:
```bash
sudo ./node.sh
```

This script creates a new user named **mpiuser** and then mount the shared NFS storage in `/home/mpiuser/shared_storage`.

## SSH Configuration

After the initial setup, it's crucial to configure SSH properly, setting up RSA public key authentication. Install the public key of `mpiuser` from the master node onto all the other nodes. This can be done by copying the public key to the `~/.ssh/authorized_keys` file of the `mpiuser` on each slave node.

## Running MPI Programs

Once the cluster is set up, you can run MPI programs. Compile the MPI test and move the resulting executable to the NFS shared directory. Then, use the `mpirun` command to run your program on the cluster:

```bash
sudo su mpiuser "mpirun -np 2 -host 192.168.254.222:2,192.168.254.225:4 /home/mpiuser/shared_storage/mpi_test.out"
```

Please note that since this is a testing setup, we are not using a configuration file for the hosts. Instead, we suggest passing the IP addresses directly in the command line. This approach is suitable for a simple setup but may not be ideal for larger clusters.

As for the external libraries, they are dynamically linked. Therefore, it is advisable to use mk modules for compatibility. This ensures that the correct versions of the libraries are loaded at runtime, avoiding potential conflicts or errors due to mismatched library versions.
