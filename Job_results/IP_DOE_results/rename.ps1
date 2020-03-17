for ($i=68; $i -ge 5; $i=$i-1 ) {
	$p = $i + 11
	$pre = "{0}" -f $i
	$digits = $i -split ''

	if ($digits.Length -eq 3) {
		$x1 = $digits[1]
		$ren = "{0}_" -f $p
		$regex = "\b[$x1]?[_]"
	
	} elseif($digits.Length -eq 4) {
		$x1 = $digits[1]
		$x2 = $digits[2]
		$ren = "{0}" -f $p
		$regex = "^[$x1][$x2]+(?=_)"
	}

	write-output $regex

	get-childitem * | Where-Object { $_.Name -match $regex } | Rename-Item -NewName { $_.Name -replace $regex, $ren }
		
		<# Move-Item -Path $_.Directory -Destination $dir #>
}


<# $files = gci -file -name 
$regex  ='^[^_]+(?=_)' 
foreach ($file in $files){ 
	$Matches = $file -match $regex 
	$servername = $Matches[0] 
	Write-Host "The server name is $Matches" 
} #>