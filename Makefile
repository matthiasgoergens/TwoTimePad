
#!/usr/bin/fish
#ghc --make Test; and for n in 1 2 3 4 5 6 7; time ./Test $n | sort | uniq -c | tee corpus{$n}alpha | sort -gr > corpus{$n}freq; echo $n ; end

all: output/corpus1freq output/corpus3freq output/corpus5freq output/corpus7freq output/corpus9freq

output/corpus%freq: output/corpus%alpha
	time sort -gr $< > $@

clean:
	rm -r output/
	mkdir output
	rm Filter Test

Filter: src/Filter.hs
	cd src; ghc -O3 --make Filter;
	ln -sf src/Filter $@

output/corpus: Filter $(wildcard data/*.txt)
	cat $(wildcard data/*.txt) | time ./Filter > $@

Test: src/Test.hs
	cd src; ghc -O3 --make Test
	ln -sf src/Test Test

output/tuples%: output/corpus Test
	time ./Test output/corpus $* > $@

output/corpus%alpha: output/tuples%
	sort $< | uniq -c > $@

output/corpus%freq: output/corpus%alpha
	time sort -gr $< > $@

.SECONDARY:
