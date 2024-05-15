**Il repository è aperto a contributi, basta aprire una [`pull request`](https://github.com/thezingaro/slurm-job-template-disi/pulls)!**

**Repository is open to contributions, please open a [`pull request`](https://github.com/thezingaro/slurm-job-template-disi/pulls)!**

## Cluster di HPC con GPU per esperimenti di calcolo (draft version 1.0)

Per poter utilizzare il cluster il primo passo è abilitare l'account istituzionale per l'accesso ai sistemi del DISI. 
Se già attivo, avrai accesso con le credenziali istituzionali, anche in remoto (`SSH`), a tutte le macchine dei *laboratori Ercolani* e *Ranzani*. 

**La quota studente massima è per ora impostata a 400 MB**. 
In caso di necessità di maggiore spazio potrai ricorrere alla creazione di una cartella in `/public/` **che viene di norma cancellata ogni prima domenica del mese**.

`/home/` utente e `/public/` sono spazi di archiviazione condivisi tra le macchine, potrai dunque creare l'ambiente di esecuzione e i file necessari all'elaborazione sulla macchina **SLURM** ([slurm.cs.unibo.it](http://slurm..cs.unibo.it)) da cui poi avviare il *job* che verrà eseguito sulle macchine dotate di GPU.

## Esecuzione di programmi C/OpenMP

Attualmente i nodi del cluster usano processori quad-core su un
singolo socket. Di conseguenza, nell'esecuzione di programmi OpenMP
sarà possibile chiedere un massimo di 4 core.

Consideriamo il programma `omp-program.c` seguente:

```C
// omp-program.c
#include <stdio.h>
#include <omp.h>

int main( void )
{
#pragma omp parallel
    printf( "Hello from core %d of %d\n",
            omp_get_thread_num(), omp_get_num_threads() );
    return 0;
}
```

Occorre innanzitutto compilarlo usando il flag `-fopenmp` del
compilatore _gcc_:

```bash
gcc -fopenmp omp-program.c -o omp-program
```

Fatto questo, è necessario creare uno script SLURM
`run-omp-program.sh` come il seguente, supponendo di volere utilizzare
quattro core:

```bash
#!/bin/bash
# run-omp-program.sh

#SBATCH --cpus-per-task 4
#SBATCH --output slurm-%j.out

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "== Running with $OMP_NUM_THREADS threads =="
srun ./omp-program
echo "== End of Job =="
```

Il parametro `--cpus-per-task 4` richiede a SLURM di allocare quattro
core su un nodo. Il job può essere inviato al sistema con il comando

```bash
sbatch run-omp-program.sh
```

al termine del quale verrà creato sulla directory corrente un file
`slurm-NNNNN.out` il cui contenuto sarà simile a

```
== Running with 4 threads ==
Hello from core 0 of 4
Hello from core 2 of 4
Hello from core 3 of 4
Hello from core 1 of 4
== End of Job ==
```

## Esecuzione di programmi C/MPI

Consideriamo il programma MPI seguente:

```C
// mpi-program.c
#include <stdio.h>
#include <mpi.h>

int main( int argc, char *argv[] )
{
	int rank, size, len;
	char hostname[MPI_MAX_PROCESSOR_NAME];
	MPI_Init( &argc, &argv );
	MPI_Comm_rank( MPI_COMM_WORLD, &rank);
	MPI_Comm_size( MPI_COMM_WORLD, &size);
	MPI_Get_processor_name( hostname, &len );
	printf( "Hello from processor %d of %d running on %s\n",
		rank, size, hostname);
	MPI_Finalize();
	return 0;
}
```

Per compilare si può usare il comando

```bash
mpicc mpi-program.c -o mpi-program
```

Per eseguire su 12 core si può usare lo script `run-omp-program.sh`
seguente:

```bash
#!/bin/bash
# run-omp-program.sh

#SBATCH -n 12
#SBATCH --output slurm-%j.out

echo "== Running with $SLURM_NTASKS tasks =="
srun --mpi=pmix_v4 -n $SLURM_NTASKS ./mpi-program
echo "== End of Job =="
```

(attenzione al parametro `---mpi=mpix_v4` che è importante per la
corretta esecuzione). Dopo aver sottomesso il job con

```
sbatch run-omp-program.sh
```

al termine dell'esecuzione verrà creato un file di output
`slurm-NNNNN.out` il cui contenuto sarà simile a

```
== Running with 12 tasks ==
Hello from processor 1 of 12 running on ws3gpu2
Hello from processor 3 of 12 running on ws3gpu2
Hello from processor 2 of 12 running on ws3gpu2
Hello from processor 0 of 12 running on ws3gpu2
Hello from processor 8 of 12 running on ws4gpu2
Hello from processor 4 of 12 running on ws4gpu1
Hello from processor 9 of 12 running on ws4gpu2
Hello from processor 10 of 12 running on ws4gpu2
Hello from processor 11 of 12 running on ws4gpu2
Hello from processor 5 of 12 running on ws4gpu1
Hello from processor 6 of 12 running on ws4gpu1
Hello from processor 7 of 12 running on ws4gpu1
== End of Job ==

```

## Esecuzione di programmi CUDA/C

Consideriamo il programma `cuda-program.cu` seguente:

```C
// cuda-program.cu
#include <stdio.h>

__global__ void mykernel(void) { }

int main(void)
{
	mykernel<<<1,1>>>( );
	printf("Hello World!\n");
	return 0;
}
```

Al momento il compilatore NVidia non è installato sul nodo frontend;
occorre quindi che nel job sia presente anche il comando di
compilazione. A tale scopo si può usare il file `run-cuda-program.sh`
seguente:

```bash
#!/bin/bash
# run-cuda-program.sh

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output slurm-%j.out

echo "=== CUDA program starts ==="
nvcc cuda-program.cu -o cuda-program && srun ./cuda-program
echo "=== End of Job ==="
```

che produrrà il solito file di input con contenuto:

```
=== CUDA program starts ===
Hello World!
=== End of Job ===
```


## Istruzioni 

Una possibile impostazione del lavoro potrebbe essere 
1. creare un virtual environment Python (ad esempio usando il comando `virtualenv`);
2. inserre all'interno del `virtualenv environment` tutto ciò di cui si ha bisogno (dati, etc.);
3. Installare altri moduli necessari con `pip` (gestori di paccheti).

### Note

Per usare **Python 3** è necessario invocarlo esplicitamente in quanto sulle macchine il default è **Python 2**. 
Nel cluster sono presenti **GPU** **Tesla** pilotate con driver `Nvidia v. 460.67` e librerie di computazione `CUDA 11.2.1`, quindi in caso di installazione di pytorch bisognerà utilizzare il comando:

```bash
pip3 install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

Il cluster utilizza uno schedulatore **SLURM** ([https://slurm.schedmd.com/overview.html](https://slurm.schedmd.com/overview.html)) per la distribuzione dei job. 
Per sottomettere un job bisogna predisporre nella propria area di lavoro un file di configurzione SLURM (nell'esempio sotto lo abbiamo nominato `script.sbatch`). 

Dopo le direttive **SLURM** è possibile inserire comandi di script (ad es. BASH). 

```bash
#!/bin/bash
#SBATCH --job-name=nomejob
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nome.cognome@unibo.it
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=nomeoutput
#SBATCH --gres=gpu:1

. bin/activate  # per attivare il virtual environment python

python test.py # per lanciare lo script python
```

Nell'esempio precedente:
- L'istruzione da tenere immutata è `--gres=gpu:1` (ogni nodo di computazione ha un'unica GPU a disposizione e deve essere attivata per poterla utilizzare). 
- Tutte le altre istruzioni di configurazione per SLURM **possono essere personalizzate**. Per la definizione di queste e altre direttive si rimanda alla documentazione ufficiale di SLURM (https://slurm.schedmd.com/sbatch.html). 
- Nell'esempio, dopo le istruzioni di configurazione di SLURM è stato invocato il programma.

Per poter avviare il job sulle macchine del cluster, è necessario:
1. accedere via SSH alla macchina [slurm.cs.unibo.it](http://slurm.cs.unibo.it) con le proprie credenziali;
2. lanciare il comando `sbatch <nomescript>`.

Alcune note importanti:
- saranno inviate e-mail per tutti gli evnti che riguardano il job lanciato, all'indirizzo specificato nelle istruzioni di configurazione (ad esempio al termine del job e nel caso di errori);
- i risultati dell'elaborazione saranno presenti nel file `<nomeoutput>` indicato nelle istruzioni di configurazioni;
- l'esecuzione sulle macchine avviene all'interno dello stesso **path relativo** che, essendo condiviso, viene visto anche dalle macchine dei laboratori e dalla macchina slurm.
