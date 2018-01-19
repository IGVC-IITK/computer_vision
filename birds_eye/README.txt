README for birds_eye node

This is a one time run node to compute transformation to top_view for a camera.

Camera view must have a chessboard.
Writes the transformation matrix in top_view.txt, which can be read by top_view node.

Command:
	rosrun birds_eye birds_eye_node [image_topic:=/input/topic] board_w board_h
	board_w, board_h --> width and height of chess board (the no. of internal corners) 