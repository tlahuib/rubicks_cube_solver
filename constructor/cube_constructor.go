package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
	"strconv"
	"strings"

	_ "github.com/lib/pq"
)

type Cube struct {
	Pieces   [3][3][3]Piece
	Moves    []string
	IsSolved bool
}
type Piece struct {
	Id       int
	ColorMap map[rune]int // Each color has a face assigned to it
}
type Rotation struct {
	Axis      int
	Direction bool
}
type Move struct {
	Axis      int
	Line      int
	Direction bool
}
type Embed struct {
	Embed []int
}

var initialCube = [][][][]rune{
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
var initialFaceColors = map[rune]int{
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
var possibleRotations = []Rotation{
	{Axis: 0, Direction: true},  //0
	{Axis: 0, Direction: false}, //1
	{Axis: 1, Direction: true},  //2
	{Axis: 1, Direction: false}, //3
	{Axis: 2, Direction: true},  //4
	{Axis: 2, Direction: false}, //5
}
var centerPieces = [6][3]int{
	{0, 1, 1}, {2, 1, 1}, {1, 1, 0},
	{1, 1, 2}, {1, 0, 1}, {1, 2, 1},
}
var cornerPieces = [8][3]int{
	{0, 0, 0}, {0, 0, 2}, {0, 2, 0}, {0, 2, 2},
	{2, 0, 0}, {2, 0, 2}, {2, 2, 0}, {2, 2, 2},
}
var standardMoves = map[[4]int][]int{
	{0, 0, 0, 0}: {}, {0, 0, 0, 2}: {5, 2}, {0, 0, 0, 4}: {4, 1},
	{0, 0, 2, 0}: {5}, {0, 0, 2, 3}: {1}, {0, 0, 2, 4}: {3, 4, 4},
	{0, 2, 0, 0}: {4}, {0, 2, 0, 2}: {2, 5, 2}, {0, 2, 0, 5}: {2},
	{0, 2, 2, 0}: {4, 4}, {0, 2, 2, 3}: {1, 4}, {0, 2, 2, 5}: {2, 5},
	{2, 0, 0, 1}: {0, 5, 2}, {2, 0, 0, 2}: {0}, {2, 0, 0, 4}: {3},
	{2, 0, 2, 1}: {0, 0}, {2, 0, 2, 3}: {5, 3}, {2, 0, 2, 4}: {3, 5},
	{2, 2, 0, 1}: {2, 2}, {2, 2, 0, 2}: {0, 4}, {2, 2, 0, 5}: {2, 4},
	{2, 2, 2, 1}: {5, 0, 0}, {2, 2, 2, 3}: {0, 3, 3}, {2, 2, 2, 5}: {0, 0, 2},
}
var flatLocations = map[[3]int]int{
	{0, 0, 0}: 0,
	{0, 0, 1}: 1,
	{0, 0, 2}: 2,
	{0, 1, 0}: 3,
	{0, 1, 2}: 4,
	{0, 2, 0}: 5,
	{0, 2, 1}: 6,
	{0, 2, 2}: 7,
	{1, 0, 0}: 8,
	{1, 0, 2}: 9,
	{1, 2, 0}: 10,
	{1, 2, 2}: 11,
	{2, 0, 0}: 12,
	{2, 0, 1}: 13,
	{2, 0, 2}: 14,
	{2, 1, 0}: 15,
	{2, 1, 2}: 16,
	{2, 2, 0}: 17,
	{2, 2, 1}: 18,
	{2, 2, 2}: 19,
}
var faceDistances = map[[2]int]int{
	{0, 0}: 0, {1, 0}: 2, {2, 0}: 1, {3, 0}: -1, {4, 0}: -1, {5, 0}: 1,
	{0, 1}: -2, {1, 1}: 0, {2, 1}: -1, {3, 1}: 1, {4, 1}: 1, {5, 1}: -1,
	{0, 2}: -1, {1, 2}: 1, {2, 2}: 0, {3, 2}: 2, {4, 2}: -1, {5, 2}: 1,
	{0, 3}: 1, {1, 3}: -1, {2, 3}: -2, {3, 3}: 0, {4, 3}: 1, {5, 3}: -1,
	{0, 4}: 1, {1, 4}: -1, {2, 4}: 1, {3, 4}: -1, {4, 4}: 0, {5, 4}: 2,
	{0, 5}: -1, {1, 5}: 1, {2, 5}: -1, {3, 5}: 1, {4, 5}: -2, {5, 5}: 0,
}

func getInitialRotations(colors []rune) map[rune]int {
	colorMap := make(map[rune]int)

	for _, color := range colors {
		colorMap[color] = initialFaceColors[color]
	}

	return colorMap
}

func initializeCube() Cube {
	// The array is face, line, column
	// The faces are ordered front, back, left, right, top, bottom
	var cube Cube

	for layer := 0; layer < 3; layer++ {
		for row := 0; row < 3; row++ {
			for col := 0; col < 3; col++ {
				colors := initialCube[layer][row][col]
				colorMap := getInitialRotations(colors)
				cube.Pieces[layer][row][col] = Piece{ColorMap: colorMap, Id: flatLocations[[3]int{layer, row, col}]}
			}
		}
	}
	cube.IsSolved = true

	return cube
}

func copyPiece(piece Piece) Piece {
	var newPiece Piece

	newPiece.ColorMap = make(map[rune]int)
	for color, location := range piece.ColorMap {
		newPiece.ColorMap[color] = location
	}

	newPiece.Id = piece.Id

	return newPiece
}

func copyCube(cube Cube) Cube {
	var newCube Cube

	for segment := 0; segment < 3; segment++ {
		for row := 0; row < 3; row++ {
			for col := 0; col < 3; col++ {
				newCube.Pieces[segment][row][col] = copyPiece(cube.Pieces[segment][row][col])
			}
		}
	}

	newCube.IsSolved = cube.IsSolved
	newCube.Moves = append(newCube.Moves, cube.Moves...)

	return newCube
}

func moveZ(cube Cube, section int, direction bool) Cube {
	var newPiece Piece
	newCube := copyCube(cube)

	for row := 0; row < 3; row++ {
		for col := 0; col < 3; col++ {
			// Select the new piece for the location
			if direction {
				newPiece = copyPiece(cube.Pieces[section][2-col][row])
			} else {
				newPiece = copyPiece(cube.Pieces[section][col][2-row])
			}

			// Rotate new piece accordingly
			for color, r := range newPiece.ColorMap {
				newPiece.ColorMap[color] = zFaces[direction][r]
			}

			// Substitute the piece
			newCube.Pieces[section][row][col] = newPiece
		}
	}

	return newCube
}

func moveY(cube Cube, col int, direction bool) Cube {
	var newPiece Piece
	newCube := copyCube(cube)

	for section := 0; section < 3; section++ {
		for row := 0; row < 3; row++ {
			// Select the new piece for the location
			if direction {
				newPiece = copyPiece(cube.Pieces[row][2-section][col])
			} else {
				newPiece = copyPiece(cube.Pieces[2-row][section][col])
			}

			// Rotate new piece accordingly
			for color, r := range newPiece.ColorMap {
				newPiece.ColorMap[color] = yFaces[direction][r]
			}

			// Substitute the piece
			newCube.Pieces[section][row][col] = newPiece
		}
	}

	return newCube
}

func moveX(cube Cube, row int, direction bool) Cube {
	var newPiece Piece
	newCube := copyCube(cube)

	for section := 0; section < 3; section++ {
		for col := 0; col < 3; col++ {
			// Select the new piece for the location
			if direction {
				newPiece = copyPiece(cube.Pieces[2-col][row][section])
			} else {
				newPiece = copyPiece(cube.Pieces[col][row][2-section])
			}

			// Rotate new piece accordingly
			for color, r := range newPiece.ColorMap {
				newPiece.ColorMap[color] = xFaces[direction][r]
			}

			// Substitute the piece
			newCube.Pieces[section][row][col] = newPiece
		}
	}

	return newCube
}

func MoveCube(cube Cube, move Move) Cube {
	var newCube Cube

	switch move.Axis {
	case 0:
		newCube = moveX(cube, move.Line, move.Direction)
	case 1:
		newCube = moveY(cube, move.Line, move.Direction)
	case 2:
		newCube = moveZ(cube, move.Line, move.Direction)
	}

	CheckSolvedCube(newCube)
	newCube.Moves = append(newCube.Moves, MoveNotation[move])

	return newCube
}

func RotateCube(cube Cube, rotation Rotation) Cube {
	newCube := copyCube(cube)

	for line := 0; line < 3; line++ {
		move := Move{Axis: rotation.Axis, Line: line, Direction: rotation.Direction}
		newCube = MoveCube(newCube, move)
	}

	return newCube
}

func standardizePosition(cube Cube) Cube {
	var location [4]int
	newCube := copyCube(cube)

	for _, coord := range cornerPieces {
		piece := cube.Pieces[coord[0]][coord[1]][coord[2]]
		if piece.Id == 0 {
			location = [4]int{coord[0], coord[1], coord[2], piece.ColorMap['w']}
			break
		}
	}

	for _, rotId := range standardMoves[location] {
		newCube = RotateCube(newCube, possibleRotations[rotId])
	}

	piece := newCube.Pieces[0][0][0]
	if (piece.Id != 0) || (piece.ColorMap['w'] != 0) {
		fmt.Println(location)
		fmt.Println(standardMoves[location])
		PrintCube(cube)

		for _, rotId := range standardMoves[location] {
			cube = RotateCube(cube, possibleRotations[rotId])
			PrintCube(cube)
		}

		PrintCube(newCube)
		panic(fmt.Sprintf("Standardization Error:\n\tID: %v\tWhite Face: %v", piece.Id, piece.ColorMap['w']))
	}

	return newCube
}

func GetPossibleMoves(cube Cube) []Cube {
	var movedCubes []Cube

	count := 0
	for newMove := range MoveNotation {
		movedCubes = append(movedCubes, MoveCube(cube, newMove))
		count++
	}

	return movedCubes
}

func GetFaceColors(cube Cube) map[rune]int {
	faceColors := make(map[rune]int)

	for _, loc := range centerPieces {
		piece := cube.Pieces[loc[0]][loc[1]][loc[2]]
		for color, r := range piece.ColorMap {
			faceColors[color] = r
		}
	}

	return faceColors
}

func GetCorrectLocation(piece Piece, faceColors map[rune]int) [3]int {
	var face int
	location := [3]int{1, 1, 1}

	for color := range piece.ColorMap {
		face = faceColors[color]

		switch face {
		case 0:
			location[0] = 0
		case 1:
			location[0] = 2
		case 2:
			location[2] = 0
		case 3:
			location[2] = 2
		case 4:
			location[1] = 0
		case 5:
			location[1] = 2
		}
	}

	return location
}

func CheckCorrectLocation(piece Piece, faceColors map[rune]int) bool {
	for color, r := range piece.ColorMap {
		if faceColors[color] != r {
			return false
		}
	}

	return true
}

func CheckSolvedCube(cube Cube) {
	faceColors := GetFaceColors(cube)

	for section := 0; section < 3; section++ {
		for row := 0; row < 3; row++ {
			for col := 0; col < 3; col++ {
				piece := cube.Pieces[section][row][col]
				if !CheckCorrectLocation(piece, faceColors) {
					cube.IsSolved = false
					return
				}
			}
		}
	}

	cube.IsSolved = true
}

func scrambleCube(cube Cube, nMoves int) Cube {
	newCube := copyCube(cube)

	for i := 0; i < nMoves; i++ {
		move := Move{rand.Intn(3), rand.Intn(3), rand.Float32() <= 0.5}
		newCube = MoveCube(newCube, move)
	}
	return newCube
}

func InitializeScrambledCube(nMoves int) Cube {
	cube := initializeCube()

	cube = scrambleCube(cube, nMoves)

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
	// 		|---|
	// 		|-4-|
	//      |---|
	// |---||---||---||---|
	// |-2-||-0-||-3-||-1-|
	// |---||---||---||---|
	// 		|---|
	// 		|-5-|
	// 		|---|
	var faces [6][3][3]rune

	for layer := 0; layer < 3; layer++ {
		for row := 0; row < 3; row++ {
			for col := 0; col < 3; col++ {
				piece := cube.Pieces[layer][row][col]
				faceMap := pieceToFace[[3]int{layer, row, col}]
				for color, r := range piece.ColorMap {
					coord := faceMap[r]
					faces[r][coord[0]][coord[1]] = color
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

func SprintCube(cube Cube) string {
	strCube := ""
	faces := CubeToFaces(cube)

	// top face
	for i := 0; i < 3; i++ {
		strCube += blank + stringifyLine(faces[4][i]) + "\\n\\n"
	}

	// left, front, right and back face
	for i := 0; i < 3; i++ {
		strCube += stringifyLine(faces[2][i]) + stringifyLine(faces[0][i]) + stringifyLine(faces[3][i]) + stringifyLine(faces[1][i]) + "\\n"
	}
	strCube += "\\n"

	// top bottomPosition
	for i := 0; i < 3; i++ {
		strCube += blank + stringifyLine(faces[5][i]) + "\\n"
	}

	return strCube
}

func EmbedCube(cube Cube) Embed {

	var embed Embed
	var faceEntropy [6]int

	stdCube := standardizePosition(cube)
	faceColors := GetFaceColors(stdCube)

	for segment := 0; segment < 3; segment++ {
		for row := 0; row < 3; row++ {
			for col := 0; col < 3; col++ {

				coord := [3]int{segment, row, col}

				// Piece & Locations
				piece := stdCube.Pieces[segment][row][col]
				correctLocation := GetCorrectLocation(piece, faceColors)

				// Calculate location distances
				for i, loc := range correctLocation {
					embed.Embed = append(embed.Embed, loc-coord[i])
				}

				// Calculate rotation distances
				for _, color := range []rune{'w', 'y', 'b', 'g', 'r', 'o'} {
					if r, ok := piece.ColorMap[color]; ok {
						embed.Embed = append(embed.Embed, faceDistances[[2]int{faceColors[color], r}])
					}
				}
			}
		}
	}

	faces := CubeToFaces(stdCube)
	for _, faceId := range faceColors {
		face := faces[faceId]
		counter := make(map[rune]int)
		for _, row := range face {
			for _, val := range row {
				counter[val] = 0
			}
		}

		faceEntropy[faceId] = len(counter)
	}

	embed.Embed = append(embed.Embed, faceEntropy[:]...)

	return embed
}

func CompareEmbeddings(origEmbed Embed, embed Embed) ([]int, []int) {
	var diff []int
	var invDiff []int

	for i, mov := range embed.Embed {
		iDiff := mov - origEmbed.Embed[i]
		diff = append(diff, iDiff)
		invDiff = append(invDiff, -iDiff)
	}

	return diff, invDiff
}

func arrayToString(a []int, delim string) string {
	return strings.Trim(strings.Replace(fmt.Sprint(a), " ", delim, -1), "[]")
}

func readConfig(file string) map[string]string {

	config := make(map[string]string)

	// Open our jsonFile
	jsonFile, err := os.Open(file)
	// if we os.Open returns an error then handle it
	if err != nil {
		panic(err)
	}
	// defer the closing of our jsonFile so that we can parse it later on
	defer jsonFile.Close()

	byteValue, _ := ioutil.ReadAll(jsonFile)

	json.Unmarshal([]byte(byteValue), &config)

	return config
}

func writeConfig(config map[string]string, file string) {
	jsonConfig, err := json.MarshalIndent(config, "", "\t")
	if err != nil {
		panic(fmt.Sprintf("Error while encoding the config file:\n\t%s", err.Error()))
	}

	err = ioutil.WriteFile(file, jsonConfig, 0644)
	if err != nil {
		panic(fmt.Sprintf("Error while writting the config file:\n\t%s", err.Error()))
	}
}

func connectDataBase(config map[string]string) *sql.DB {

	psqlInfo := fmt.Sprintf(
		"host=%s port=%s user=%s password=%s dbname=%s sslmode=disable",
		config["host"], config["port"], config["user"], config["pass"], config["db"],
	)

	db, err := sql.Open("postgres", psqlInfo)
	if err != nil {
		panic(err)
	}

	err = db.Ping()
	if err != nil {
		panic(err)
	}

	fmt.Printf("Database %v succesfully connected\n", config["db"])

	return db
}

func insertCubeToDB(config map[string]string, db *sql.DB, embed Embed, strCube string, nMoves int) {

	sqlStatement := fmt.Sprintf(
		`INSERT INTO %s.%s (abs_embedding, representation, min_moves) VALUES(ARRAY [%s], E'%s', %v)`,
		config["schema"], config["table"], arrayToString(embed.Embed[:], ","), strCube, nMoves,
	)

	_, err := db.Exec(sqlStatement)
	if err != nil {
		if err.Error() != `pq: duplicate key value violates unique constraint "cubes_pkey"` {
			panic(fmt.Sprintf("%s:\n\tARRAY [%s]", err.Error(), arrayToString(embed.Embed[:], ",")))
		}
	}
}

func insertEmbedToDB(config map[string]string, db *sql.DB, origEmbed Embed, embed Embed, moveDiff int) {
	diffEmbed, invDiffEmbed := CompareEmbeddings(origEmbed, embed)

	sqlStatement := "INSERT INTO %s.differences (embed_diff, move_diff) VALUES(ARRAY [%s], %v)"

	_, err := db.Exec(fmt.Sprintf(sqlStatement, config["schema"], arrayToString(diffEmbed, ","), moveDiff))
	if err != nil {
		panic(fmt.Sprintf("%s:\n\tARRAY [%s]", err.Error(), arrayToString(diffEmbed, ",")))
	}

	_, err = db.Exec(fmt.Sprintf(sqlStatement, config["schema"], arrayToString(invDiffEmbed, ","), -moveDiff))
	if err != nil {
		panic(fmt.Sprintf("%s:\n\tARRAY [%s]", err.Error(), arrayToString(invDiffEmbed, ",")))
	}
}

func getCubeMoves(config map[string]string, db *sql.DB, embed Embed, currentMoves int) int {
	var nMoves int

	sqlStatement := fmt.Sprintf(
		"SELECT min_moves FROM %s.%s WHERE abs_embedding = ARRAY [%v]",
		config["schema"], config["table"], arrayToString(embed.Embed[:], ","),
	)

	rows, err := db.Query(sqlStatement)
	if err != nil {
		panic(fmt.Sprintf("Error while reading ARRAY [%s]:\n\t%s", arrayToString(embed.Embed[:], ","), err.Error()))
	}

	defer rows.Close()
	rows.Next()
	err = rows.Scan(&nMoves)
	if err != nil {
		if err.Error() == "sql: Rows are closed" {
			return currentMoves
		} else {
			panic(err)
		}
	}

	return nMoves
}

func getDiff(config map[string]string, db *sql.DB, diff []int) int {
	var nMoves int

	sqlStatement := fmt.Sprintf(
		"SELECT move_diff FROM %s.differences WHERE embed_diff = ARRAY [%v]",
		config["schema"], arrayToString(diff[:], ","),
	)

	rows, err := db.Query(sqlStatement)
	if err != nil {
		panic(fmt.Sprintf("Error while reading ARRAY [%s]:\n\t%s", arrayToString(diff[:], ","), err.Error()))
	}

	defer rows.Close()
	rows.Next()
	err = rows.Scan(&nMoves)
	if err != nil {
		panic(err)
	}

	return nMoves
}

func buildQueue(nMoves int) []Cube {

	queue := []Cube{initializeCube()}

	for i := 0; i < nMoves; i++ {
		buffer := []Cube{}
		for _, cube := range queue {
			buffer = append(buffer, GetPossibleMoves(cube)...)
		}
		queue = buffer
	}

	return queue

}

func exploreCubes(directory string) {

	// Read config file
	config := readConfig(directory + "config.json")
	lastCompletedState, err := strconv.Atoi(config["completed"])
	if err != nil {
		fmt.Printf("Error during 'completed state' conversion in config file: %v", config["completed"])
	}

	// Connect to db
	db := connectDataBase(config)
	defer db.Close()

	// Check for first value
	if lastCompletedState < 0 {
		insertCubeToDB(config, db, EmbedCube(initializeCube()), SprintCube(initializeCube()), 0)
		lastCompletedState = 0
	}

	queue := buildQueue(lastCompletedState)

	for {
		buffer := []Cube{}

		for _, cube := range queue {
			origEmbed := EmbedCube(cube)
			origMoves := getCubeMoves(config, db, origEmbed, lastCompletedState)
			possibleMoves := GetPossibleMoves(cube)
			buffer = append(buffer, possibleMoves...)

			for _, newCube := range possibleMoves {
				embed := EmbedCube(newCube)
				nMoves := getCubeMoves(config, db, embed, lastCompletedState+1)

				if math.Abs(float64(nMoves-origMoves)) > 1 {
					// TODO:
					// - Look for extra info within the embedding
					fmt.Println("\n\n----------------------------")
					fmt.Println(origMoves)
					fmt.Println(arrayToString(origEmbed.Embed[:], ","))
					PrintCube(cube)
					fmt.Println("\n\n")
					fmt.Println(nMoves)
					fmt.Println(embed.Embed[:], ",")

					PrintCube(standardizePosition(newCube))
					panic(fmt.Sprintf("Difference (%v) is greater than |1|", nMoves-origMoves))
				}
				insertCubeToDB(config, db, embed, SprintCube(newCube), nMoves)
				insertEmbedToDB(config, db, origEmbed, embed, nMoves-origMoves)
			}
		}

		fmt.Printf("--- Completed for [%v] moves ---\n", lastCompletedState)
		lastCompletedState++
		config["completed"] = strconv.Itoa(lastCompletedState)
		writeConfig(config, directory+"config.json")
		queue = buffer
	}
}

func main() {
	exploreCubes("solves/v7/")
}
