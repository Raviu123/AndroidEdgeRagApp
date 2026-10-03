$xml = [xml](Get-Content "app/build/test-results/testDebugUnitTest/TEST-com.siddu.offlinedocai.GenericGroundedSynthesisTest.xml")
$test = $xml.testsuite.testcase | Where-Object { $_.name -eq 'testRealDoc2_List_TechnicalManagementProcesses' }
Write-Output $test.'system-out'
