package main

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
	"sync"

	"github.com/rubik"
)

type DBCube struct {
	Embed          []int
	Representation string
	minMoves       int
}
type DBDiff struct {
	EmbedDiff []int
	MoveDiff  int
}

var waitProcess sync.WaitGroup

func jsonCube(cube rubik.Cube) []byte {
	for i, s := range cube.Moves {
		cube.Moves[i] = strings.ReplaceAll(s, "'", "''")
	}

	b, err := json.Marshal(cube)
	if err != nil {
		panic(err)
	}

	return b
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

	byteValue, _ := io.ReadAll(jsonFile)

	json.Unmarshal([]byte(byteValue), &config)

	return config
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

func buildQueue(db *sql.DB) {
	var hasData bool

	rows, err := db.Query("select exists (select * from rubik.cubes q limit 1)")
	if err != nil {
		panic(err)
	}

	rows.Next()
	err = rows.Scan(&hasData)
	if err != nil {
		panic(err)
	}
	rows.Close()

	if !hasData {
		cube := rubik.InitializeCube()
		addCubeToQueue(db, cube)
		fmt.Println("---Initialized Queue---")
		dbCube := DBCube{Embed: rubik.EmbedCube(cube), Representation: rubik.SprintCube(cube)}
		insertCubeToDB(db, dbCube)
	}
}

func addCubeToQueue(db *sql.DB, cube rubik.Cube) {

	b := jsonCube(cube)

	sqlStatement := fmt.Sprintf(`INSERT INTO rubik.queue (cube, n_moves) VALUES('%s', %v)`, string(b), len(cube.Moves))

	_, err := db.Exec(sqlStatement)
	if err != nil {
		fmt.Println(sqlStatement)
		panic(err)
	}
}

func readQueue(db *sql.DB, nCubes int) []rubik.Cube {
	var id int
	var oldId int
	var ids []int
	var cubes []rubik.Cube
	var nMoves int

	sqlStatement := fmt.Sprintf(
		"select * from rubik.queue order by id limit %v",
		nCubes,
	)

	rows, err := db.Query(sqlStatement)
	if err != nil {
		panic(err)
	}

	first := true
	count := 0
	for rows.Next() {
		var b []byte
		var newMoves int
		var cube rubik.Cube

		err = rows.Scan(&id, &b, &newMoves)
		if err != nil {
			panic(err)
		}

		ids = append(ids, id)
		if first {
			nMoves = newMoves
			first = false
		} else {
			if oldId > id {
				panic(fmt.Sprintf("Error: Old Id (%v) is not greater than the new Id (%v)", id, oldId))
			}
		}

		oldId = id

		if newMoves != nMoves {
			fmt.Printf("Shifting from %v moves to %v moves\n", nMoves, newMoves)
			break
		}

		err = json.Unmarshal(b, &cube)
		if err != nil {
			panic(err)
		}

		cubes = append(cubes, cube)

		count++
	}
	rows.Close()

	sqlStatement = fmt.Sprintf("DELETE FROM rubik.queue where id in (%s)", arrayToString(ids, ", "))

	_, err = db.Exec(sqlStatement)
	if err != nil {
		panic(err)
	}

	return cubes
}

func insertCubeToDB(db *sql.DB, dbCube DBCube) {

	sqlStatement := fmt.Sprintf(
		`INSERT INTO rubik.cubes (abs_embedding, representation, min_moves) VALUES(ARRAY [%s], E'%s', %v)`,
		arrayToString(dbCube.Embed[:], ","), dbCube.Representation, dbCube.minMoves,
	)

	_, err := db.Exec(sqlStatement)
	if err != nil {
		if err.Error() != `pq: duplicate key value violates unique constraint "cubes_pkey"` {
			panic(fmt.Sprintf("%s:\n\tARRAY [%s]", err.Error(), arrayToString(dbCube.Embed[:], ",")))
		}
	}

}

func insertEmbedToDB(db *sql.DB, dbDiff DBDiff) {

	sqlStatement := "INSERT INTO rubik.differences (embed_diff, move_diff) VALUES(ARRAY [%s], %v)"

	_, err := db.Exec(fmt.Sprintf(sqlStatement, arrayToString(dbDiff.EmbedDiff, ","), dbDiff.MoveDiff))
	if err != nil {
		panic(fmt.Sprintf("%s:\n\tARRAY [%s]", err.Error(), arrayToString(dbDiff.EmbedDiff, ",")))
	}

}

func getCubeMoves(db *sql.DB, embed []int, currentMoves int) int {
	var nMoves int

	sqlStatement := fmt.Sprintf(
		"SELECT min_moves FROM rubik.cubes WHERE abs_embedding = ARRAY [%v]",
		arrayToString(embed[:], ","),
	)

	rows, err := db.Query(sqlStatement)
	if err != nil {
		panic(fmt.Sprintf("Error while reading ARRAY [%s]:\n\t%s", arrayToString(embed[:], ","), err.Error()))
	}

	rows.Next()
	err = rows.Scan(&nMoves)
	if err != nil {
		if err.Error() == "sql: Rows are closed" {
			return currentMoves
		} else {
			panic(err)
		}
	}
	rows.Close()

	return nMoves
}

func calculateDiff(db *sql.DB, origEmbed []int, origMoves int, cube rubik.Cube, cCube chan DBCube, cDiff chan DBDiff) {

	// Calculate new embedding and send it for writting
	newEmbed := rubik.EmbedCube(cube)
	minMoves := getCubeMoves(db, newEmbed, origMoves+1)
	cCube <- DBCube{Embed: rubik.EmbedCube(cube), Representation: rubik.SprintCube(cube), minMoves: minMoves}

	// Calculate differences
	diffEmbed := rubik.CompareEmbeddings(origEmbed, newEmbed)

	diffEmbed = append(origEmbed, diffEmbed...)

	if math.Abs(float64(minMoves-origMoves)) > 1 {
		fmt.Println(origEmbed)
		fmt.Println(newEmbed)
		panic(fmt.Sprintf("Error, move difference greater than one (%v, %v)", origMoves, minMoves))
	}

	// Send differences for writting
	cDiff <- DBDiff{EmbedDiff: diffEmbed, MoveDiff: minMoves - origMoves}

}

func processCube(db *sql.DB, cube rubik.Cube, cCube chan DBCube, cDiff chan DBDiff, cQueue chan rubik.Cube) {

	origEmbed := rubik.EmbedCube(cube)
	origMoves := getCubeMoves(db, origEmbed, -1)
	if origMoves == -1 {
		fmt.Println(cube.Moves)
		rubik.PrintCube(cube)
		panic(fmt.Sprintf("Error: Cube previously processed not found\n\t%s", arrayToString(origEmbed, ",")))
	}

	possibleMoves := rubik.GetPossibleMoves(cube)
	for _, newCube := range possibleMoves {
		cQueue <- newCube
		go calculateDiff(db, origEmbed, origMoves, newCube, cCube, cDiff)
	}

}

func gatherEmbeds(db *sql.DB, c chan DBCube) {

	for {
		dbCube, more := <-c
		if !more {
			return
		}

		insertCubeToDB(db, dbCube)

		waitProcess.Done()
	}

}

func gatherDiffs(db *sql.DB, c chan DBDiff) {

	for {
		dbDiff, more := <-c
		if !more {
			return
		}

		insertEmbedToDB(db, dbDiff)

		waitProcess.Done()
	}

}

func gatherQueue(db *sql.DB, c chan rubik.Cube) {

	for {
		cube, more := <-c
		if !more {
			return
		}

		addCubeToQueue(db, cube)

		waitProcess.Done()
	}

}

func exploreCubes(directory string, n_parallel int) {

	// Read config file
	config := readConfig(directory + "config.json")

	// Connect to db
	db := connectDataBase(config)
	defer db.Close()

	// Check for first value
	buildQueue(db)

	// Process by batches
	for {

		// Initialize processes for writting
		cCubes := make(chan DBCube)
		cDiff := make(chan DBDiff)
		cQueue := make(chan rubik.Cube)
		go gatherEmbeds(db, cCubes)
		go gatherDiffs(db, cDiff)
		go gatherQueue(db, cQueue)

		// Read queue
		buffer := readQueue(db, n_parallel)

		waitProcess.Add(len(buffer) * (len(rubik.MoveNotation) * 3))
		// Launch parallel processing
		for _, cube := range buffer {
			go processCube(db, cube, cCubes, cDiff, cQueue)
		}

		waitProcess.Wait()
		close(cCubes)
		close(cDiff)
		close(cQueue)
	}

}

func main() {
	directory := "solves/v7/"
	exploreCubes(directory, 10)
}
