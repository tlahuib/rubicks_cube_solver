package main

import (
	"fmt"
	"math/rand"
)

type Cube struct {
	// Cube     [6][3][3]rune
	Pieces   [3][3][3]Piece
	IsSolved bool
}
type Piece struct {
	Colors          []rune
	CorrectLocation [3]int
	Rotation        []int // Same shape as colors. Each color references a face (0, 5)
}
type Move struct {
	Axis      int
	Line      int
	Direction bool
}

var solvedCube = [][][][]rune{
	{
		{{'w', 'b', 'r'}, {'w', 'r'}, {'w', 'g', 'r'}},
		{{'w', 'b'}, {'w'}, {'w', 'g'}},
		{{'w', 'b', 'o'}, {'w', 'o'}, {'w', 'g', 'o'}},
	}, {
		{{'b', 'r'}, {'r'}, {'g', 'r'}},
		{{'b'}, {}, {'g'}},
		{{'b', 'o'}, {'o'}, {'g', 'o'}},
	}, {
		{{'y', 'b', 'r'}, {'y', 'r'}, {'y', 'g', 'r'}},
		{{'y', 'b'}, {'y'}, {'y', 'g'}},
		{{'y', 'b', 'o'}, {'y', 'o'}, {'y', 'g', 'o'}},
	},
}
var faceColors = map[rune]int{
	'w': 0,
	'y': 1,
	'b': 2,
	'g': 3,
	'r': 4,
	'o': 5,
}
var pieceToFace = map[[3]int]map[int][2]int{
	{0, 0, 0}: {0: {0, 0}, 2: {0, 2}, 4: {2, 0}},
	{0, 0, 1}: {0: {0, 1}, 4: {2, 1}},
	{0, 0, 2}: {0: {0, 2}, 3: {0, 0}, 4: {2, 2}},
	{0, 1, 0}: {0: {1, 0}, 2: {1, 2}},
	{0, 1, 1}: {0: {1, 1}},
	{0, 1, 2}: {0: {1, 2}, 3: {1, 0}},
	{0, 2, 0}: {0: {2, 0}, 2: {2, 2}, 5: {0, 0}},
	{0, 2, 1}: {0: {2, 1}, 5: {0, 1}},
	{0, 2, 2}: {0: {2, 2}, 3: {2, 0}, 5: {0, 2}},
	{1, 0, 0}: {2: {0, 1}, 4: {1, 0}},
	{1, 0, 1}: {4: {1, 1}},
	{1, 0, 2}: {3: {0, 1}, 4: {1, 2}},
	{1, 1, 0}: {2: {1, 1}},
	{1, 1, 1}: {},
	{1, 1, 2}: {3: {1, 1}},
	{1, 2, 0}: {2: {2, 1}, 5: {1, 0}},
	{1, 2, 1}: {5: {1, 1}},
	{1, 2, 2}: {3: {2, 1}, 5: {1, 2}},
	{2, 0, 0}: {2: {0, 0}, 4: {0, 0}, 1: {0, 2}},
	{2, 0, 1}: {4: {0, 1}, 1: {0, 1}},
	{2, 0, 2}: {1: {0, 0}, 3: {0, 2}, 4: {0, 2}},
	{2, 1, 0}: {2: {1, 0}, 1: {1, 2}},
	{2, 1, 1}: {1: {1, 1}},
	{2, 1, 2}: {1: {1, 0}, 3: {1, 2}},
	{2, 2, 0}: {2: {2, 0}, 1: {2, 2}, 5: {2, 0}},
	{2, 2, 1}: {5: {2, 1}, 1: {2, 1}},
	{2, 2, 2}: {1: {2, 0}, 3: {2, 2}, 5: {2, 2}},
}
var zFaces = map[bool][6]int{
	true:  {0, 1, 4, 5, 3, 2},
	false: {0, 1, 5, 4, 2, 3},
}
var yFaces = map[bool][6]int{
	true:  {4, 5, 2, 3, 1, 0},
	false: {5, 4, 2, 3, 0, 1},
}
var xFaces = map[bool][6]int{
	true:  {3, 2, 0, 1, 4, 5},
	false: {2, 3, 1, 0, 4, 5},
}
var blank string = "         "
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

func getInitialRotations(colors []rune) []int {
	var rotations []int

	for _, color := range colors {
		rotations = append(rotations, faceColors[color])
	}

	return rotations
}

func initializeCube() Cube {
	// The array is face, line, column
	// The faces are ordered front, back, left, right, top, bottom
	var cube Cube

	for layer := 0; layer < 3; layer++ {
		for row := 0; row < 3; row++ {
			for col := 0; col < 3; col++ {
				colors := solvedCube[layer][row][col]
				location := [3]int{layer, row, col}
				rotation := getInitialRotations(colors)
				cube.Pieces[layer][row][col] = Piece{Colors: colors, CorrectLocation: location, Rotation: rotation}
			}
		}
	}
	cube.IsSolved = true

	return cube
}

func moveZ(cube Cube, section int, direction bool) Cube {
	var newPiece Piece
	_cube := cube

	for row := 0; row < 3; row++ {
		for col := 0; col < 3; col++ {
			// Select the new piece for the location
			if direction {
				newPiece = cube.Pieces[section][2-col][row]
			} else {
				newPiece = cube.Pieces[section][col][2-row]
			}

			// Rotate new piece accordingly
			for i, r := range newPiece.Rotation {
				newPiece.Rotation[i] = zFaces[direction][r]
			}

			// Substitute the piece
			_cube.Pieces[section][row][col] = newPiece
		}
	}

	return _cube
}

func moveY(cube Cube, col int, direction bool) Cube {
	var newPiece Piece
	_cube := cube

	for section := 0; section < 3; section++ {
		for row := 0; row < 3; row++ {
			// Select the new piece for the location
			if direction {
				newPiece = cube.Pieces[row][2-section][col]
			} else {
				newPiece = cube.Pieces[2-row][section][col]
			}

			// Rotate new piece accordingly
			for i, r := range newPiece.Rotation {
				newPiece.Rotation[i] = yFaces[direction][r]
			}

			// Substitute the piece
			_cube.Pieces[section][row][col] = newPiece
			// cube, moves := scrambleCube(cube, 5)
		}
	}

	return _cube
}

func moveX(cube Cube, row int, direction bool) Cube {
	var newPiece Piece
	_cube := cube

	for section := 0; section < 3; section++ {
		for col := 0; col < 3; col++ {
			// Select the new piece for the location
			if direction {
				newPiece = cube.Pieces[2-col][row][section]
			} else {
				newPiece = cube.Pieces[col][row][2-section]
			}

			// Rotate new piece accordingly
			for i, r := range newPiece.Rotation {
				newPiece.Rotation[i] = xFaces[direction][r]
			}

			// Substitute the piece
			_cube.Pieces[section][row][col] = newPiece
		}
	}

	return _cube
}

func MoveCube(cube Cube, move Move) Cube {
	var _cube Cube

	switch move.Axis {
	case 0:
		_cube = moveX(cube, move.Line, move.Direction)
	case 1:
		_cube = moveY(cube, move.Line, move.Direction)
	case 2:
		_cube = moveZ(cube, move.Line, move.Direction)
	}

	_cube = CheckSolvedCube(_cube)
	return _cube
}

func CheckSolvedCube(cube Cube) Cube {
	_cube := cube
	for section := 0; section < 3; section++ {
		for row := 0; row < 3; row++ {
			for col := 0; col < 3; col++ {
				piece := cube.Pieces[section][row][col]
				if [3]int{section, row, col} != piece.CorrectLocation {
					_cube.IsSolved = false
					return _cube
				}
			}
		}
	}

	_cube.IsSolved = true
	return _cube
}

func scrambleCube(cube Cube, nMoves int) (Cube, []string) {
	var moves []string
	_cube := cube

	for i := 0; i < nMoves; i++ {
		move := Move{rand.Intn(3), rand.Intn(3), rand.Float32() <= 0.5}
		moves = append(moves, MoveNotation[move])
		_cube = MoveCube(_cube, move)
	}
	return _cube, moves
}

func InitializeScrambledCube(nMoves int) Cube {
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

func CubeToFaces(cube Cube) [6][3][3]rune {
	var faces [6][3][3]rune

	for layer := 0; layer < 3; layer++ {
		for row := 0; row < 3; row++ {
			for col := 0; col < 3; col++ {
				piece := cube.Pieces[layer][row][col]
				faceMap := pieceToFace[[3]int{layer, row, col}]
				for i, faceColor := range piece.Colors {
					face := piece.Rotation[i]
					coord := faceMap[face]
					faces[face][coord[0]][coord[1]] = faceColor
				}
			}
		}
	}

	return faces
}

func PrintCube(cube Cube) {

	faces := CubeToFaces(cube)

	// top face
	for i := 0; i < 3; i++ {
		fmt.Println(blank, stringifyLine(faces[4][i]))
	}
	fmt.Println()

	// left, front, right and back face
	for i := 0; i < 3; i++ {
		fmt.Println(
			stringifyLine(faces[2][i]),
			stringifyLine(faces[0][i]),
			stringifyLine(faces[3][i]),
			stringifyLine(faces[1][i]),
		)
	}
	fmt.Println()

	// top bottomPosition
	for i := 0; i < 3; i++ {
		fmt.Println(blank, stringifyLine(faces[5][i]))
	}
}

// func GetFacePositions(cube Cube) map[rune]int {
// 	// Output order is 'w', 'y', 'g', 'b', 'r', 'o'
// 	// Face order is:
// 	//      |---|
// 	//      |-4-|
// 	//      |---|
// 	// |---||---||---||---|
// 	// |-2-||-0-||-3-||-1-|
// 	// |---||---||---||---|
// 	//      |---|
// 	//      |-5-|
// 	//      |---|

// 	positions := make(map[rune]int)
// 	for i := 0; i < 6; i++ {
// 		positions[cube.Cube[i][1][1]] = i
// 	}

// 	return positions
// } x y z M E S
// 	positions := GetFacePositions(cube)
// 	for _, pos := range positions {
// 		for row := 0; row < 3; row++ {
// 			for col := 0; col < 3; col++ {fmt.Println(cube)
// 				tilePosition = positions[cube.Cube[pos][row][col]]
// 				embed[pos*9+row*3+col] = faceDistances[[2]int{pos, tilePosition}]
// 			}
// 		}
// 	}

// 	return embed
// }

// func scrambleEmbed(steps int, c chan []int) {
// 	for i := 0; i < steps; i++ { x y z M E S
// 		nMoves := rand.Intn(25) + 1
// 		cube := InitializeScrambledCube(nMoves)
// 		embed := EmbedCube(cube)

// 		c <- append([]int{nMoves}, embed[:]...)
// 	}
// }

// func main() {
// 	var joined []int
// 	var newMoves int
// 	var embed [54]int
// 	var solves [55]string

// 	// Check for saved solves
// 	fRead, errRead := os.Open("solves.csv")
// 	if errRead != nil {
// 		log.Fatal(errRead)fmt.Println(cube)
// 	}

// 	bestSolves := make(map[[54]int]int)fmt.Println(cube)
// 		if err == io.EOF {
// 			break
// 		}
// 		if err != nil {
// 			log.Fatal(err)
// 		}
// 		for i := 0; i < 54; i++ {
// 			embed[i], _ = strconv.Atoi(rec[i])
// 		}
// 		newMoves, _ = strconv.Atoi(rec[54])

// 		bestSolves[embed] = newMoves
// 	}
// 	fRead.Close()

// 	// Create new solves
// 	c := make(chan []int, 100)
// 	steps := 10000000= moveX(cube, 2, false)
// 				bestSolves[embed] = newMoves
// 			}
// 		} else {
// 			bestSolves[embed] = newMoves
// 		}
// 	}

// 	// Write
// 	fWrite, errWrite := os.Create("solves.csv")
// 	if errWrite != nil {
// 		log.Fatal(errWrite)
// 	}
// 	csvwriter := csv.NewWriter(fWrite)
// 	defer csvwriter.Flush()

// 	for embed, nMoves := range bestSolves {
// 		for i, val := range embed {
// 			solves[i] = strconv.Itoa(val)
// 		}
// 		solves[54] = strconv.Itoa(nMoves)

// 		csvwriter.Write(solves[:])
// 	}

// }

func main() {
	cube := initializeCube()

	PrintCube(cube)
}
