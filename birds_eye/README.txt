README for birds_eye node

This is a one time run node to compute transformation to top_view for a camera.

Camera view must have a chessboard (made of 3.5cm squares). 
Writes the transformation matrix in top_view.txt, which can be read by top_view node.

Command:
	rosrun birds_eye birds_eye_node 7 6 

	7, 6 --> width and height of chess board (the no. of internal corners) 