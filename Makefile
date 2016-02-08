test: matrix.c test.c 
	gcc -o test matrix.c test.c 

clean: 
	rm *.o; rm test; rm *.pyc
