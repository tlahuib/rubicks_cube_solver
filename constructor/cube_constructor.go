package main

import (
	"fmt"
)

var faceColors = [6]rune{'w', 'y', 'g', 'b', 'r', 'o'}
var blank string = "         "
var axisRelation = [3][4]int8{
	{0, 3, 1, 2}, // axis x, moves front, right, back and left faces
	{0, 4, 1, 5}, // axis y, moves front, top, back and bottom faces
	{2, 4, 3, 5}, // axis z, moves left, top, right and bottom faces
}

func InitializeCube() [6][3][3]rune {
	// The array is face, line, column
	// The faces are ordered front, back, left, right, top, bottom
	var cube [6][3][3]rune

	// for i := 0; i < 6; i++ {
	// 	for j := 0; j < 3; j++ {
	// 		for k := 0; k < 3; k++ {
	// 			cube[i][j][k] = faceColors[i]
	// 		}
	// 	}
	// }

	for i := 0; i < 6; i++ {
		cube[i] = [3][3]rune{
			{'1', '2', '3'},
			{'8', '0', '4'},
			{'7', '6', '5'},
		}
	}

	return cube
}

func rotateFace(cube [6][3][3]rune, face int, direction bool) [6][3][3]rune {
	_cube := cube
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			if direction {
				_cube[face][j][2-i] = cube[face][i][j]
			} else {
				_cube[face][2-j][i] = cube[face][i][j]
			}
		}
	}
	return _cube
}

func reverseCuve(cube [6][3][3]rune) [6][3][3]rune {
	var reverseCuve [6][3][3]rune

	for i := 0; i < 6; i++ {
		for j := 0; j < 3; j++ {
			for k := 0; k < 3; k++ {
				reverseCuve[i][j][k] = cube[i][2-j][2-k]
			}
		}
	}

	return reverseCuve
}

func rotateX(cube [6][3][3]rune, line int8, direction bool) [6][3][3]rune {
	var ii [4]int
	var shift int
	_cube := cube

	// The direction changes loop and shift
	if direction {
		ii = [4]int{3, 2, 1, 0}
		shift = 3
	} else {
		ii = [4]int{0, 1, 2, 3}
		shift = 1
	}

	// Iterate over rows
	for _, i := range ii {
		_cube[axisRelation[0][i]][line] = cube[axisRelation[0][(i+shift)%4]][line]
	}

	// Rotate top or bottom face
	switch line {
	case 0:
		_cube = rotateFace(_cube, 4, !direction)
	case 2:
		_cube = rotateFace(_cube, 5, direction)
	}

	return _cube
}

func rotateY(cube [6][3][3]rune, line int8, direction bool) [6][3][3]rune {
	var ii [4]int
	var shift int
	var iPrev int
	var sourceCube [6][3][3]rune
	var _line int8
	_cube := cube

	// Account for the back being reversed
	rev := reverseCuve(cube)

	// The direction changes loop and shift
	if direction {
		ii = [4]int{3, 2, 1, 0}
		shift = 3
		iPrev = 4
	} else {
		ii = [4]int{0, 1, 2, 3}
		shift = 1
		iPrev = 2
	}

	for _, i := range ii { // Iterate over faces
		if i != 2 {
			if i == iPrev {
				sourceCube = rev
			} else {
				sourceCube = cube
			}
			for j := 0; j < 3; j++ { // Iterate over Rows
				_cube[axisRelation[1][i]][j][line] = sourceCube[axisRelation[1][(i+shift)%4]][j][line]
			}
		} else {
			switch line {
			case 0:
				_line = 2
			case 1:
				_line = 1
			case 2:
				_line = 0
			}
			for j := 0; j < 3; j++ { // Iterate over Rows
				_cube[axisRelation[1][i]][2-j][_line] = cube[axisRelation[1][(i+shift)%4]][j][line]
			}
		}
	}

	// Rotate left or right face
	switch line {
	case 0:
		_cube = rotateFace(_cube, 2, !direction)
	case 2:
		_cube = rotateFace(_cube, 3, direction)
	}

	return _cube
}

func rotateZ(cube [6][3][3]rune, line int8, direction bool) [6][3][3]rune {
	var ii [4]int
	var shift int
	var _line int8
	_cube := cube

	// The direction changes loop and shift
	if direction {
		ii = [4]int{3, 2, 1, 0}
		shift = 3
	} else {
		ii = [4]int{0, 1, 2, 3}
		shift = 1
	}

	for _, i := range ii { // Iterate over faces
		// Depending on the face, the line can either top/bottom, left/right
		if (i%3 == 0) && (line != 1) {
			if line == 0 {
				_line = 2
			} else {
				_line = 0
			}
		} else {
			_line = line
		}

		// Depending on the face, line can refer to either rows or columns
		if i%2 == 0 {
			for j := 0; j < 3; j++ { // Iterate over elements
				_cube[axisRelation[2][i]][_line][j] = cube[axisRelation[2][(i+shift)%4]][j][_line]
			}
		} else {
			for j := 0; j < 3; j++ { // Iterate over elements
				_cube[axisRelation[2][i]][j][_line] = cube[axisRelation[2][(i+shift)%4]][_line][j]
			}
		}
	}

	// Rotate front or back face
	switch line {
	case 0:
		_cube = rotateFace(_cube, 0, direction)
	case 2:
		_cube = rotateFace(_cube, 1, direction)
	}

	return _cube
}

// func MoveCube(cube [6][3][3]rune, axis int8, line int8, direction bool) [6][3][3]rune {
// 	axisRel := axisRelation[axis]
// 	if direction{
// 		cube[axisRel[0]][][], cube[axisRel[0]][][], cube[axisRel[0]][][], cube[axisRel[0]][][] =

// 	}
// }

func stringifyLine(line [3]rune) string {
	var strLine string
	for i := 0; i < 3; i++ {
		strLine += "|" + string(line[i]) + "|"
	}
	return strLine
}

func PrintCube(cube [6][3][3]rune) {
	// top face
	for i := 0; i < 3; i++ {
		fmt.Println(blank, stringifyLine(cube[4][i]))
	}
	fmt.Println()

	// left, front, right and back face
	for i := 0; i < 3; i++ {
		fmt.Println(
			stringifyLine(cube[2][i]),
			stringifyLine(cube[0][i]),
			stringifyLine(cube[3][i]),
			stringifyLine(cube[1][i]),
		)
	}
	fmt.Println()

	// top bottom
	for i := 0; i < 3; i++ {
		fmt.Println(blank, stringifyLine(cube[5][i]))
	}
}

func main() {
	cube := InitializeCube()
	cube = reverseCuve(cube)

	// cube = rotateY(cube, 2, true)
	PrintCube(cube)
	// cube = rotateY(cube, 2, false)
	// fmt.Println("-------------------------------------------------------------")
	// PrintCube(cube)
}
