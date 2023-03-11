package Cube

import (
	"fmt"
	"math/rand"
)

type Move struct {
	Axis      int
	Line      int
	Direction bool
}

var faceColors = [6]rune{'w', 'y', 'g', 'b', 'r', 'o'}
var blank string = "         "
var AxisRelation = [3][4]int8{
	{0, 3, 1, 2}, // Axis x, moves front, right, back and left faces
	{0, 4, 1, 5}, // Axis y, moves front, top, back and bottom faces
	{2, 4, 3, 5}, // Axis z, moves left, top, right and bottom faces
}
var MoveNotation = map[Move]string{
	{Axis: 0, Line: 0, Direction: false}: "U",
	{Axis: 0, Line: 0, Direction: true}:  "U'",
	{Axis: 0, Line: 1, Direction: false}: "E'",
	{Axis: 0, Line: 1, Direction: true}:  "E",
	{Axis: 0, Line: 2, Direction: false}: "D'",
	{Axis: 0, Line: 2, Direction: true}:  "D",
	{Axis: 1, Line: 0, Direction: false}: "L",
	{Axis: 1, Line: 0, Direction: true}:  "L'",
	{Axis: 1, Line: 1, Direction: false}: "M",
	{Axis: 1, Line: 1, Direction: true}:  "M'",
	{Axis: 1, Line: 2, Direction: false}: "R'",
	{Axis: 1, Line: 2, Direction: true}:  "R",
	{Axis: 2, Line: 0, Direction: false}: "F'",
	{Axis: 2, Line: 0, Direction: true}:  "F",
	{Axis: 2, Line: 1, Direction: false}: "S'",
	{Axis: 2, Line: 1, Direction: true}:  "S",
	{Axis: 2, Line: 2, Direction: false}: "B",
	{Axis: 2, Line: 2, Direction: true}:  "B'",
}

func initializeCube() [6][3][3]rune {
	// The array is face, line, column
	// The faces are ordered front, back, left, right, top, bottom
	var cube [6][3][3]rune

	for i := 0; i < 6; i++ {
		for j := 0; j < 3; j++ {
			for k := 0; k < 3; k++ {
				cube[i][j][k] = faceColors[i]
			}
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

func reverseRows(face [3][3]rune) [3][3]rune {
	reversedRows := face

	reversedRows[0], reversedRows[2] = reversedRows[2], reversedRows[0]

	return reversedRows
}

func reverseCols(face [3][3]rune) [3][3]rune {
	reversedCols := face

	for i := 0; i < 3; i++ {
		reversedCols[i][0], reversedCols[i][2] = reversedCols[i][2], reversedCols[i][0]
	}

	return reversedCols
}

func reverseFace(face [3][3]rune) [3][3]rune {
	reversedFace := face

	reversedFace = reverseCols(reversedFace)
	reversedFace = reverseRows(reversedFace)

	return reversedFace
}

func rotateX(cube [6][3][3]rune, move Move) [6][3][3]rune {
	var ii [4]int
	var shift int
	_cube := cube

	// The direction changes loop and shift
	if move.Direction {
		ii = [4]int{3, 2, 1, 0}
		shift = 3
	} else {
		ii = [4]int{0, 1, 2, 3}
		shift = 1
	}

	// Iterate over rows
	for _, i := range ii {
		_cube[AxisRelation[0][i]][move.Line] = cube[AxisRelation[0][(i+shift)%4]][move.Line]
	}

	// Rotate top or bottom face
	switch move.Line {
	case 0:
		_cube = rotateFace(_cube, 4, !move.Direction)
	case 2:
		_cube = rotateFace(_cube, 5, move.Direction)
	}

	return _cube
}

func rotateY(cube [6][3][3]rune, move Move) [6][3][3]rune {
	var ii [4]int
	var shift int
	_cube := cube

	// Account for the back being reversed
	_cube[1] = reverseFace(_cube[1])
	referenceCube := _cube

	// The direction changes loop and shift
	if move.Direction {
		ii = [4]int{3, 2, 1, 0}
		shift = 3
	} else {
		ii = [4]int{0, 1, 2, 3}
		shift = 1
	}

	for _, i := range ii { // Iterate over faces
		for j := 0; j < 3; j++ { // Iterate over Rows
			_cube[AxisRelation[1][i]][j][move.Line] = referenceCube[AxisRelation[1][(i+shift)%4]][j][move.Line]
		}
	}

	// Undo the reversion
	_cube[1] = reverseFace(_cube[1])

	// Rotate left or right face
	switch move.Line {
	case 0:
		_cube = rotateFace(_cube, 2, !move.Direction)
	case 2:
		_cube = rotateFace(_cube, 3, move.Direction)
	}

	return _cube
}

func rotateZ(cube [6][3][3]rune, move Move) [6][3][3]rune {
	var ii [4]int
	var shift int
	_cube := cube

	// Account for rows/columns reversion
	_cube[2] = reverseFace(_cube[2])
	_cube[4] = reverseRows(_cube[4])
	_cube[5] = reverseCols(_cube[5])
	referenceCube := _cube

	// The direction changes loop and shift
	if move.Direction {
		ii = [4]int{3, 2, 1, 0}
		shift = 3
	} else {
		ii = [4]int{0, 1, 2, 3}
		shift = 1
	}

	for _, i := range ii { // Iterate over faces
		if i%2 == 0 { // Left and right use columns, up and bottom use rows
			for j := 0; j < 3; j++ { // Iterate over elements
				_cube[AxisRelation[2][i]][j][move.Line] = referenceCube[AxisRelation[2][(i+shift)%4]][move.Line][j]
			}
		} else {
			for j := 0; j < 3; j++ { // Iterate over elements
				_cube[AxisRelation[2][i]][move.Line][j] = referenceCube[AxisRelation[2][(i+shift)%4]][j][move.Line]
			}
		}
	}

	// Undo the reversion
	_cube[2] = reverseFace(_cube[2])
	_cube[4] = reverseRows(_cube[4])
	_cube[5] = reverseCols(_cube[5])

	// Rotate front or back face
	switch move.Line {
	case 0:
		_cube = rotateFace(_cube, 0, move.Direction)
	case 2:
		_cube = rotateFace(_cube, 1, !move.Direction)
	}

	return _cube
}

func MoveCube(cube [6][3][3]rune, move Move) [6][3][3]rune {
	var _cube [6][3][3]rune

	switch move.Axis {
	case 0:
		_cube = rotateX(cube, move)
	case 1:
		_cube = rotateY(cube, move)
	case 2:
		_cube = rotateZ(cube, move)
	}

	return _cube
}

func scrambleCube(cube [6][3][3]rune, nMoves int) ([6][3][3]rune, []string) {
	var moves []string
	_cube := cube

	for i := 0; i < nMoves; i++ {
		move := Move{rand.Intn(3), rand.Intn(3), rand.Float32() <= 0.5}
		moves = append(moves, MoveNotation[move])
		_cube = MoveCube(_cube, move)
	}

	return _cube, moves
}

func InitializeScrambledCube(nMoves int) [6][3][3]rune {
	cube := initializeCube()

	cube, _ = scrambleCube(cube, nMoves)

	return cube
}

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
