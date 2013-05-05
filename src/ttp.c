#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>
#include <string.h>

// void *mmap(void *addr, size_t length, int prot, int flags,
//                  int fd, off_t offset);

const int alpha_size = 46;
const char* const alphabet = " abcdefghijklmnopqrstuvwxyz0123456789.?,-:;'()";
const int order = 4;

// typedef short int16;
typedef struct {
  char* start;
  int length;
} range;

#define handle_error(msg) \
    do { perror(msg); exit(EXIT_FAILURE); } while (0)

range* my_mmap (char* filename) {
    struct stat sb;

    int fd = open(filename, O_RDONLY);
    if (fd == -1)
        handle_error("open");

    if (fstat(fd, &sb) == -1)           /* To obtain file size */
        handle_error("fstat");

    char* addr = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED)
        handle_error("mmap");
    range *r = (range*) malloc (sizeof(range));
    r->start = addr;
    r->length = sb.st_size;

    // (range { start = char, length = sb.st_size});
    return r;
}

char* toTable = NULL;
char* fromTable = NULL;
char toITA (char input) {
  assert (toTable != NULL);
  char output = toTable[(int) input];
  assert (output < alpha_size);
  return output; }
char fromITA (char input) {
  assert (fromTable != NULL);
  assert (input < alpha_size);
  char output = fromTable[(int) input];
  assert (output != 0);
  return output; }


void buildTables () {
  toTable = calloc (256, sizeof(char));
  fromTable = malloc (alpha_size * sizeof(char));
  for (int i = 0; i < 256; i++) {
    fromTable[i] = alpha_size; }
  const char* alpha = alphabet; int i=0;
  for (; *alpha; i++, alpha++) {
    toTable[(int) *alpha] = i;
    fromTable[i] = *alpha; }
  printf ("Done building.\n"); }


int** getCorpus (char* filename) {
  range* r = my_mmap (filename);
  int** t = (int**) malloc(order * sizeof(int*));
  char* end = r->start + r->length;
  printf ("Done.\n");
  for (int i = 0, size=1; i < order; i++) {
    size*=alpha_size;
    printf ("size: %i\n", size);
    t[i] = (int*) calloc(size, sizeof(int));
    printf ("calloced."); }
  printf ("Done 2.\n");
  for (char* s = r->start; s < end; s++) {
    printf ("corpus: %c\n", *s);
    int size = 1;
    for (int i = 0, pos = 0; (i < order) && (s+i < end); i++) {
      size*=alpha_size;      
      pos = alpha_size * pos + toITA (*(s+i));
      printf ("size: %i, pos: %i\n", size, pos);
      t[i][pos] += 1;
    } }
  return t; }

void mapToITA (range r) {
  range* r_ = (range*) malloc (sizeof(range));
  r_ -> length = r.length+1;
  r_ -> start = (char*) calloc (r.length + 1, sizeof(char));
  for (int i = 0; i < r.length; i++) {
    r_->start[i] = toITA (i); }
  // return r_;
}

void buildFreqs () {
  range* r = my_mmap ("corpus1freq");
  int lineLength = strchr (r->start, '\n') - r->start + 1;
  printf ("%i\n", r->length % lineLength);
  printf ("%i\n", lineLength);
  for (int i = 0; i*lineLength < r->length; i++) {
    printf ("%i ", *(r->start + (lineLength)*i - 1));
  }
  printf ("\n");
  //printf("second line: %i\n", strchr (r->start + 48, '\n') - r->start);
}
/*   for (int i=1;i <= 7; i++) { */
/*     int fd = open(filename, O_RDONLY); */
    
/*   } */
/* } */
/*   int** freqn; */
/*   char** freqa; */

/*   int** alphan; */
/*   char** freqa; */
  
/* } */

/* test_mm () { */
/*   range* r = my_mmap ("corpus5freq"); */
/*   strchr("\n" */
/* } */

int main () {
  buildTables ();
  buildFreqs ();
  // assert (sizeof(int16) >= 2);
  // printf("%s %i %i\n", bla+3, (bla+4) < (bla+s), s);
  // getCorpus ("corpus");
  /* int i; */
  /* char* s = malloc (200 * sizeof(char)); */
  /* sscanf ("     121 \"1212\"","%i \"%s\"",&i,s); */
  /* printf ("%i %s\n", i, s); */
  return 0;
}
