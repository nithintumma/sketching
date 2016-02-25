RM = rm -f
FW = -framework Accelerate 

sketch: matrix.c svd.c sketch.c
	gcc -o sketch -Ofast -g -framework Accelerate matrix.c svd.c svd_lapack.c sketch.c

test: matrix.c test.c 
	gcc -o test matrix.c test.c 

svd: matrix.c svd.c
	gcc -o svd matrix.c svd.c

ssvd: matrix.c sample_svd.c
	gcc -o ssvd matrix.c sample_svd.c

lsvd: matrix.c svd_lapack.c
	gcc -o lsvd -Ofast -g -framework Accelerate matrix.c svd_lapack.c 

clean: 
	$(RM) *.o; $(RM) test; $(RM) svd; $(RM) ssvd; $(RM) sketch; $(RM) lsvd; $(RM) *.pyc
