## After checking the matrices in utils/models/BLG
The matrix M is correct. For the right hopping matrix Tx, it should actually be Tx.T , and then it would be completely correct. The way it is written right now, the matrix Tx represents hopping to the closest neighbor in the -x direction, but then this should come with a $$ e^{+ik_x} $$ instead.

For the right hopping matrix Ty, the story is the same as Tx. It should actually be just the transpose of what is written there. 
Just to clarify, as it is written now, the 'right hopping matrix' would Ty.conj().T, but this would have the sign of kx wrong in the exponential. 

## Resolution: 
I believe if I do a new run of the LDOS with the matrices as 
M -> M.T and 
Ty -> Ty.T , 
then everything should be correct. Starting a run now.

## Does this work? Seemingly yes. The LDOS is exactly correct. 
This directory is to do a bigger run on alice
