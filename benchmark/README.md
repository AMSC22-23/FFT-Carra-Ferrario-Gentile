`benchmark.sh` run a benchmark and format the output in a table-format thanks to print_table.sh. 

Source it:
```bash
$ source print_table.sh
```

Run it:
```bash
$ printTable "," "$(./benchmark.sh ../build/test_*.out <MPI number of processes> <Strategy header 1> <Strategy header 2>)"
``` 
OMP example:
```bash
$ printTable "," "$(./benchmark.sh ../build/test_OMP.out OMP SEQ)"
``` 
MPI example:
```bash
$ printTable "," "$(./benchmark.sh ../build/test_MPI.out 4 MPI SEQ)"
``` 
OMP Output:

  ```
  +     +              +              +              +              +             +             +        +
  | n   | OMP forward  | OMP inverse  | SEQ forward  | SEQ inverse  | SU forward  | SU inverse  | error  |
  +     +              +              +              +              +             +             +        +
  | 14  | 5.03684 ms   | 2.01387 ms   | 5.4197 ms    | 4.31228 ms   | 1.07601     | 2.14129     | 0      |
  | 15  | 9.14538 ms   | 3.74544 ms   | 6.19548 ms   | 5.51005 ms   | 0.677444    | 1.47113     | 0      |
  | 16  | 17.4198 ms   | 28.4487 ms   | 15.3593 ms   | 13.7089 ms   | 0.881713    | 0.481881    | 0      |
  | 17  | 43.728 ms    | 69.4294 ms   | 29.4753 ms   | 30.8734 ms   | 0.674061    | 0.444673    | 0      |
  | 18  | 41.209 ms    | 46.7397 ms   | 66.3256 ms   | 62.9481 ms   | 1.60949     | 1.34678     | 0      |
  | 19  | 76.4807 ms   | 110.004 ms   | 132.191 ms   | 150.465 ms   | 1.72842     | 1.36781     | 0      |
  | 20  | 170.788 ms   | 210.589 ms   | 269.416 ms   | 297.57 ms    | 1.57749     | 1.41304     | 0      |
  | 21  | 376.108 ms   | 383.665 ms   | 565.101 ms   | 607.972 ms   | 1.5025      | 1.58465     | 0      |
  | 22  | 726.752 ms   | 796.857 ms   | 1.18943 s    | 1.27607 s    | 1.63664     | 1.60138     | 0      |
  | 23  | 1.46138 s    | 1.6076 s     | 2.47462 s    | 2.6395 s     | 1.69335     | 1.64189     | 0      |
  | 24  | 3.15642 s    | 3.30362 s    | 5.18471 s    | 5.51811 s    | 1.64259     | 1.67032     | 0      |
  | 25  | 6.76 s       | 6.93919 s    | 10.9497 s    | 11.2511 s    | 1.61978     | 1.62138     | 0      |
  +     +              +              +              +              +             +             +        +  
  ```