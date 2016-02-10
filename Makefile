RM = rm -f

test: matrix.c test.c 
	gcc -o test matrix.c test.c 

svd: matrix.c svd.c
	gcc -o svd matrix.c svd.c

ssvd: matrix.c sample_svd.c
	gcc -o ssvd matrix.c sample_svd.c

clean: 
	$(RM) *.o; $(RM) test; $(RM) svd; $(RM) ssvd; $(RM) *.pyc
